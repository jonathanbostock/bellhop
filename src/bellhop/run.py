"""``run()`` — the one-shot pipeline: a library replacement for run.sh.

create -> wait-functional -> upload codebase -> setup+run (tee'd) -> pull
results -> upload to GCS -> teardown -> structured RunResult.

Backend-agnostic: hand it a ``PodConfig`` (RunPod), ``ModalConfig`` (Modal),
``LambdaConfig`` (Lambda Cloud) or ``NebiusConfig`` (Nebius) and the pipeline
runs identically — the only provider-specific work happens behind
:func:`bellhop.backend.open_box`. (Every backend guarantees a writable
``/workspace``, so the ``/workspace/<slug>`` layout is uniform.)
"""

from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .backend import open_box
from .errors import (
    ExecTimeoutError,
    GcsUploadError,
    PreflightError,
    RemoteJobError,
    ResultsMissingError,
)

if TYPE_CHECKING:
    from .lambda_box import LambdaConfig
    from .modal_box import ModalConfig
    from .nebius_box import NebiusConfig
    from .pod import PodConfig

    Backend = PodConfig | ModalConfig | LambdaConfig | NebiusConfig

# GCS upload is opt-in: pass gcs_base="gs://your-bucket/prefix" (or --gcs-base) to enable.
DEFAULT_GCS_BASE = None


@dataclass
class RunSpec:
    slug: str
    codebase: str                       # local dir OR git URL (auto-detected)
    run: str                            # the job (required)
    setup: str | None = None            # deps, run before `run`
    results_subdir: str = "results"     # path on the pod to pull back
    local_out: str | None = None        # default ./experiments/<slug>
    gcs_base: str | None = DEFAULT_GCS_BASE   # set None to skip GCS upload
    env: dict[str, str] = field(default_factory=dict)
    # optional client-side cap (seconds) on the job step; None (default) lets
    # the job run until the box's server-side TTL (max_lifetime etc.) kills it
    timeout: float | None = None


@dataclass
class RunResult:
    slug: str
    pod_id: str                         # box id (RunPod pod id or Modal sandbox id)
    remote_exit: int
    local_results: str
    gcs_uri: str | None
    retrieve_cmd: str | None
    log_tail: str


def _is_git(codebase: str) -> bool:
    return codebase.startswith(("http://", "https://", "git@"))


def _job_script(spec: RunSpec, run_dir: str) -> str:
    """The setup+run script, tee'd to run.log.

    ``set -e`` inside the block makes a failing ``setup`` abort the job instead
    of silently running against a half-configured box (and applies to ``run``
    too: the first failing command decides the exit status).
    """
    setup_block = f"echo '--- setup ---'\n{spec.setup}\n" if spec.setup else ""
    return (
        f"cd {shlex.quote(run_dir)}\n"
        f"mkdir -p {shlex.quote(spec.results_subdir)}\n"
        f"{{\nset -e\n{setup_block}echo '--- run ---'\n{spec.run}\n}} 2>&1 | tee {shlex.quote(spec.results_subdir + '/run.log')}\n"
        f"exit ${{PIPESTATUS[0]}}\n"
    )


async def _checked_exec(box, cmd: str, what: str) -> None:
    r = await box.exec(cmd)
    if r.exit_code != 0:
        raise RemoteJobError(f"{what} failed", remote_exit=r.exit_code,
                             log_tail=r.stderr[-2000:])


async def run(spec: RunSpec, backend: "Backend", *, keep_pod: bool = False,
              api_key: str | None = None) -> RunResult:
    """Run ``spec`` on the box implied by ``backend``'s config type.

    ``keep_pod`` leaves the box up after the run (kept for name compatibility;
    applies to every backend — NB on Lambda/Nebius a kept box has NO server-side
    TTL). ``api_key`` is the RunPod key and is ignored by the other backends
    (they use their own ambient auth).
    """
    if not (spec.slug and spec.codebase and spec.run):
        raise PreflightError("slug, codebase and run are all required")
    if not _is_git(spec.codebase) and not Path(spec.codebase).is_dir():
        raise PreflightError(f"codebase dir not found: {spec.codebase}")

    local_out = spec.local_out or os.path.join(os.getcwd(), "experiments", spec.slug)
    Path(local_out).mkdir(parents=True, exist_ok=True)
    run_dir = f"/workspace/{spec.slug}"
    results_remote = f"{run_dir}/{spec.results_subdir}"
    # Don't mutate the caller's config (run_many shares one across the sweep) —
    # give each run its own per-slug name.
    backend = replace(backend, name=f"bellhop-{spec.slug}")

    async with open_box(backend, keep=keep_pod, api_key=api_key) as p:
        # --- upload codebase (mkdir -p the parent so both git-clone and push
        # work even when /workspace doesn't pre-exist, e.g. on a Modal image) ---
        await _checked_exec(p, f"mkdir -p {shlex.quote(os.path.dirname(run_dir))}",
                            "workspace setup (mkdir)")
        if _is_git(spec.codebase):
            r = await p.exec(f"git clone --depth 1 {shlex.quote(spec.codebase)} {shlex.quote(run_dir)}")
            if r.exit_code != 0:
                raise RemoteJobError("git clone failed", remote_exit=r.exit_code, log_tail=r.stderr[-2000:])
        else:
            await _checked_exec(p, f"mkdir -p {shlex.quote(run_dir)}",
                                "workspace setup (mkdir)")
            await p.push(spec.codebase, run_dir)

        # --- run (setup then job), tee'd to a log that travels back ---
        timed_out: ExecTimeoutError | None = None
        try:
            job_res = await p.exec(_job_script(spec, run_dir), env=spec.env,
                                   timeout=spec.timeout)
            remote_exit = job_res.exit_code
        except ExecTimeoutError as e:
            # Still try to salvage whatever the job wrote before re-raising —
            # partial results + run.log are exactly what you want after a
            # timeout, and the box is torn down on exit either way.
            timed_out = e
            remote_exit = None

        # --- pull results ---
        if await p.exists_remote(results_remote):
            await p.pull(results_remote, local_out)
        elif remote_exit == 0:
            raise ResultsMissingError(f"job succeeded but no results dir at {results_remote}")

        if timed_out is not None:
            raise timed_out

        # --- upload to GCS (from this box; creds never touch the pod) ---
        gcs_uri = retrieve_cmd = None
        if spec.gcs_base:
            gcs_uri = spec.gcs_base.rstrip("/") + f"/{spec.slug}/"
            await _gcs_upload(local_out, gcs_uri)
            retrieve_cmd = f"gcloud storage cp -r {gcs_uri} ./"

        # pull() extracts to local_out/<basename(results_remote)> — use the same
        # derivation so a nested results_subdir ("out/results") still resolves.
        pulled_dir = os.path.basename(results_remote.rstrip("/"))
        log_tail = _tail(os.path.join(local_out, pulled_dir, "run.log"))
        result = RunResult(
            slug=spec.slug, pod_id=p.id, remote_exit=remote_exit,
            local_results=local_out, gcs_uri=gcs_uri, retrieve_cmd=retrieve_cmd, log_tail=log_tail,
        )

    if remote_exit != 0:
        raise RemoteJobError(f"remote job exited {remote_exit}", remote_exit=remote_exit, log_tail=result.log_tail)
    return result


async def run_many(specs: list[RunSpec], backend: "Backend", *,
                   max_concurrency: int = 4, **kw) -> list[RunResult | BaseException]:
    """Fan a sweep out across boxes. Returns results/exceptions positionally.

    The same ``backend`` config is shared across the sweep; ``run`` copies it
    per-slug, so concurrent runs don't clobber each other's name.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(s: RunSpec):
        async with sem:
            return await run(s, backend, **kw)

    return await asyncio.gather(*(_one(s) for s in specs), return_exceptions=True)


async def _gcs_upload(local_dir: str, gcs_uri: str) -> None:
    entries = list(Path(local_dir).iterdir())
    if not entries:
        return
    proc = await asyncio.create_subprocess_exec(
        "gcloud", "storage", "cp", "-r", *[str(e) for e in entries], gcs_uri,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise GcsUploadError(f"gcloud upload failed: {err.decode('utf-8','replace')[:500]}")


def _tail(path: str, n: int = 20) -> str:
    try:
        return "\n".join(Path(path).read_text("utf-8", "replace").splitlines()[-n:])
    except Exception:
        return ""
