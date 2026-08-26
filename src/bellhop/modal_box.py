"""The Modal Sandbox backend — the Modal-side analogue of :mod:`bellhop.pod`.

A Modal Sandbox is an ephemeral container you drive imperatively, which maps
almost 1:1 onto bellhop's box lifecycle:

    create  ->  exec / push / pull  ->  terminate

Two things are *simpler* than the RunPod path and so are absent here:

- **No readiness probe.** When ``Sandbox.create`` returns, the box is execable;
  there's no sshd-lags-RUNNING window to poll past.
- **No GraphQL TTL dance.** Modal exposes native server-side timers as plain
  ``create`` kwargs: ``timeout`` (hard max lifetime, ~= RunPod ``terminate_after``)
  and ``idle_timeout`` (terminate after inactivity, ~= ``stop_after``).

``modal`` is an optional dependency (``pip install 'bellhop-py[modal]'``); it is
imported lazily so a RunPod-only install never needs it. Code/result transfer is
tar-over-exec (only needs ``tar`` in the image), mirroring the pod's tar-over-ssh.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator

from .backend import TAR_EXCLUDES, ExecResult
from .errors import ExecTimeoutError, PreflightError, ProvisionError


def _import_modal():
    try:
        import modal
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise PreflightError(
            "the Modal backend needs the `modal` package — install it with "
            "`pip install 'bellhop-py[modal]'`"
        ) from e
    return modal


@dataclass
class ModalConfig:
    """Config for a Modal Sandbox box. The Modal-side peer of ``PodConfig``."""

    gpu: str | None = None                 # Modal GPU vocab, e.g. "A10G", "A100", "H100", "T4", "L4"; None = CPU
    image: Any = None                      # a modal.Image, a registry string, or None -> default
    image_preset: str | None = None        # key into the builders below (debian-slim, pytorch-cuda)
    pip: list[str] = field(default_factory=list)   # convenience installs onto the resolved image
    apt: list[str] = field(default_factory=list)
    cpu: float | None = None               # cores (Modal default if None)
    memory: int | None = None              # MiB
    workdir: str | None = None             # default container workdir; run() uses absolute paths regardless
    region: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    secrets: list = field(default_factory=list)    # modal.Secret objects (creds never go through argv)
    volumes: dict = field(default_factory=dict)    # {mount_path: modal.Volume}
    app_name: str = "bellhop"              # Modal app the sandbox is anchored to (created if missing)
    name: str = "bellhop"                  # logical label (matches PodConfig.name; run() sets per-slug)
    # native server-side TTL (plain create kwargs; survive host death)
    timeout: timedelta | None = timedelta(hours=24)   # hard max lifetime
    idle_timeout: timedelta | None = None             # terminate after this much inactivity
    # unified spelling of the hard kill, same name as PodConfig.max_lifetime;
    # wins over timeout when set.
    max_lifetime: timedelta | None = None

    def __post_init__(self):
        if self.max_lifetime is not None:
            self.timeout = self.max_lifetime

    def resolve_image(self):
        """Build the modal.Image for this config (lazy-imports modal)."""
        modal = _import_modal()
        if isinstance(self.image, modal.Image):
            return self._with_extras(self.image, ensure_base=False)
        if isinstance(self.image, str):
            return self._with_extras(modal.Image.from_registry(self.image), ensure_base=False)
        if self.image_preset:
            return self._with_extras(_preset_image(modal, self.image_preset), ensure_base=True)
        return self._with_extras(modal.Image.debian_slim(), ensure_base=True)

    def _with_extras(self, img, *, ensure_base: bool):
        # tar is required for push/pull; git is needed for git-URL codebases.
        # We only inject onto images we control (default / preset) and trust
        # user-supplied images/registry refs to already have them.
        if ensure_base:
            img = img.apt_install("git", "tar")
        if self.apt:
            img = img.apt_install(*self.apt)
        if self.pip:
            img = img.pip_install(*self.pip)
        return img


_PRESETS = ("debian-slim", "pytorch-cuda")


def _preset_image(modal, key: str):
    if key == "debian-slim":
        return modal.Image.debian_slim()
    if key == "pytorch-cuda":
        # torch 2.4.0 + CUDA 12.4 — kept in lockstep with the RunPod preset of
        # the same name (pod.IMAGE_PRESETS) so the key means the same
        # environment on either backend.
        return modal.Image.from_registry(
            "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
        )
    raise PreflightError(f"unknown modal image_preset {key!r} (have {list(_PRESETS)})")


def _create_kwargs(config: ModalConfig, *, image=None, app=None) -> dict:
    """Assemble Sandbox.create kwargs. Pure (no network) so it's unit-testable."""
    kw: dict = {"app": app, "image": image}
    if config.gpu:
        kw["gpu"] = config.gpu
    if config.cpu is not None:
        kw["cpu"] = config.cpu
    if config.memory is not None:
        kw["memory"] = config.memory
    if config.workdir:
        kw["workdir"] = config.workdir
    if config.region:
        kw["region"] = config.region
    if config.env:
        kw["env"] = dict(config.env)
    if config.secrets:
        kw["secrets"] = list(config.secrets)
    if config.volumes:
        kw["volumes"] = dict(config.volumes)
    if config.timeout:
        kw["timeout"] = int(config.timeout.total_seconds())
    else:
        # There is no "unlimited" on Modal: with the kwarg omitted, Modal applies
        # its own default (300s). That is NOT the PodConfig None-means-no-TTL
        # semantic, so make the 5-minute box impossible to hit by surprise.
        warnings.warn(
            "ModalConfig timeout=None does not disable the sandbox lifetime — "
            "Modal always enforces one (its default is 300s); set timeout= or "
            "max_lifetime= explicitly for anything longer than a smoke test",
            stacklevel=2,
        )
    if config.idle_timeout:
        kw["idle_timeout"] = int(config.idle_timeout.total_seconds())
    return kw


class Sandbox:
    """A live Modal Sandbox. Construct via :func:`sandbox` (the async CM).

    Satisfies the :class:`bellhop.backend.ExecBox` protocol, so ``run()`` drives
    it through the exact same calls it uses for a RunPod ``Pod``.
    """

    def __init__(self, sb, config: ModalConfig):
        self._sb = sb
        self.config = config
        self.id = sb.object_id

    async def exec(self, cmd: str, env: dict[str, str] | None = None,
                   timeout: float | None = None) -> ExecResult:
        """Run command(s) in the sandbox.

        No client-side timeout by default — the sandbox's own TTL
        (``timeout``/``max_lifetime`` on the config) is the backstop; pass a
        finite ``timeout`` (seconds) to cap this one command via Modal's
        native per-exec timeout. On expiry this raises
        :class:`~bellhop.errors.ExecTimeoutError`, matching the pod backend
        (the ExecBox contract).

        Env is passed natively (over Modal's API, not argv) so secret values
        never appear in the container's process list — the same guarantee the
        pod path gets by feeding the script over stdin.
        """
        script = f"set -o pipefail\n{cmd}\n"
        start = time.monotonic()
        try:
            proc = await self._sb.exec.aio(
                "bash", "-c", script,
                env=dict(env or {}),
                timeout=math.ceil(timeout) if timeout is not None else None,
            )
            out = await proc.stdout.read.aio()
            err = await proc.stderr.read.aio()
            code = await proc.wait.aio()
        except Exception as e:
            if timeout is not None and "timeout" in type(e).__name__.lower():
                raise self._timeout_err(cmd, timeout) from e
            raise
        # Modal enforces the native timeout by killing the process and
        # reporting a plain non-zero exit; translate that into the typed error.
        if timeout is not None and code != 0 and time.monotonic() - start >= timeout:
            raise self._timeout_err(cmd, timeout)
        return ExecResult(code or 0, out, err)

    def _timeout_err(self, cmd: str, timeout: float) -> ExecTimeoutError:
        head = cmd.strip().splitlines()[0][:120] if cmd.strip() else cmd
        return ExecTimeoutError(
            f"exec timed out after {timeout:.0f}s in sandbox {self.id}: {head}")

    async def push(self, local: str | Path, remote: str) -> None:
        """Upload a local directory to ``remote`` (tar-over-exec).

        The archive is buffered in memory; fine for codebases, and large
        artifacts belong in a Volume / GCS rather than a push anyway.
        """
        import shlex

        local = str(local)
        if not Path(local).is_dir():
            raise PreflightError(f"push source not a directory: {local}")
        tar = await asyncio.create_subprocess_exec(
            "tar", "czf", "-", "-C", local, *TAR_EXCLUDES, ".",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        data, terr = await tar.communicate()
        if tar.returncode != 0:
            raise RuntimeError(f"push: local tar failed: {terr.decode('utf-8', 'replace')[:500]}")
        remote_cmd = f"mkdir -p {shlex.quote(remote)} && tar xzf - -C {shlex.quote(remote)}"
        proc = await self._sb.exec.aio("bash", "-c", remote_cmd, text=False)
        proc.stdin.write(data)
        proc.stdin.write_eof()
        await proc.stdin.drain.aio()
        code = await proc.wait.aio()
        if code != 0:
            err = (await proc.stderr.read.aio()) or b""
            raise RuntimeError(f"push: remote untar failed (rc={code}): {err.decode('utf-8', 'replace')[:500]}")

    async def pull(self, remote: str, local_dest: str | Path) -> None:
        """Download remote dir into ``local_dest`` (creates local_dest/<basename>)."""
        import os
        import shlex

        local_dest = str(local_dest)
        Path(local_dest).mkdir(parents=True, exist_ok=True)
        parent = os.path.dirname(remote.rstrip("/")) or "/"
        base = os.path.basename(remote.rstrip("/"))
        remote_cmd = f"tar czf - -C {shlex.quote(parent)} {shlex.quote(base)}"
        proc = await self._sb.exec.aio("bash", "-c", remote_cmd, text=False)
        data = await proc.stdout.read.aio()
        code = await proc.wait.aio()
        if code != 0:
            err = (await proc.stderr.read.aio()) or b""
            raise RuntimeError(f"pull: remote tar failed (rc={code}): {err.decode('utf-8', 'replace')[:500]}")
        untar = await asyncio.create_subprocess_exec(
            "tar", "xzf", "-", "-C", local_dest,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, uerr = await untar.communicate(data)
        if untar.returncode != 0:
            raise RuntimeError(f"pull: local untar failed: {uerr.decode('utf-8', 'replace')[:500]}")

    async def exists_remote(self, path: str) -> bool:
        import shlex

        proc = await self._sb.exec.aio("bash", "-c", f"test -e {shlex.quote(path)}")
        return (await proc.wait.aio()) == 0

    async def call(self, fn, *args, **kwargs):
        """Run a local Python function in the sandbox; see :func:`bellhop.call.call`."""
        from .call import call as _call
        return await _call(self, fn, *args, **kwargs)

    async def teardown(self) -> None:
        await self._sb.terminate.aio()


@contextlib.asynccontextmanager
async def sandbox(config: ModalConfig, *, keep: bool = False) -> AsyncIterator[Sandbox]:
    """Provision a Modal Sandbox, yield it, terminate it.

    On any exception the sandbox is still terminated, unless ``keep=True``. No
    readiness wait is needed — ``Sandbox.create`` returns an execable box.
    """
    modal = _import_modal()
    image = config.resolve_image()
    app = await modal.App.lookup.aio(config.app_name, create_if_missing=True)
    kwargs = _create_kwargs(config, image=image, app=app)
    try:
        sb = await modal.Sandbox.create.aio(**kwargs)
    except Exception as e:
        raise ProvisionError(f"modal sandbox create failed: {e}") from e

    box = Sandbox(sb, config)
    try:
        yield box
    finally:
        if not keep:
            with contextlib.suppress(Exception):
                await box.teardown()
