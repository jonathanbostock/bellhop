"""Shared SSH transport for VM-shaped backends (RunPod, Lambda, Nebius).

A RunPod pod, a Lambda Cloud instance, and a Nebius VM are all the same thing
once they're up: an SSH-able Linux box. This module holds that common half —
exec (stdin-fed script, no secrets in argv), tar-over-ssh push/pull, and the
raw probe channel — so each backend only implements what actually differs:
how the box is created, how its SSH endpoint is discovered, and how it dies.

A backend subclasses :class:`SshBox` and provides:

- ``id`` — the provider's box id (for error messages),
- ``ssh_user`` — the login user (``root`` on RunPod, ``ubuntu`` elsewhere),
- ``_ssh_key`` — path to the private key,
- ``_ssh_endpoint()`` — ``(host, port)``, raising
  :class:`~bellhop.errors.PodNotReadyError` while unroutable.

Everything here predates the multi-provider split — it is the Pod transport
from pod.py, moved verbatim so behaviour (and the offline tests that
monkeypatch it) carry over unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

from .backend import TAR_EXCLUDES, ExecResult
from .errors import ExecTimeoutError, PodNotReadyError, PreflightError, ProvisionError

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=30",
]


def resolve_ssh_key(explicit: str | None) -> str:
    """The private-key path a config asks for (default ``~/.ssh/id_ed25519``)."""
    key = explicit or os.path.expanduser("~/.ssh/id_ed25519")
    if not Path(key).exists():
        raise PreflightError(f"ssh private key not found: {key}")
    return key


def pubkey_text(explicit: str | None) -> str:
    pub = resolve_ssh_key(explicit) + ".pub"
    if not Path(pub).exists():
        raise PreflightError(f"ssh public key not found: {pub}")
    return Path(pub).read_text().strip()


class SshBox:
    """The transport half of an ExecBox, for backends reached over SSH."""

    id: str
    ssh_user: str
    _ssh_key: str
    _noun = "box"          # how error messages name the box ("pod", "instance")
    # Flip to True mid-session to convert this box to keep=True — the CM's
    # teardown honors it. This is what makes failure-aware keeps composable:
    # run(keep_on_failure=True) sets it when a job dies holding state, and
    # user code can set it on any condition ("keep if the checkpoint marker
    # exists") without threading a flag through the context manager.
    keep = False
    # Optional grace hook the lifetime watchdog awaits (bounded) before it
    # tears the box down — run() points it at an emergency results pull.
    on_lifetime_expiry = None

    def _ssh_endpoint(self) -> tuple[str, int]:
        raise NotImplementedError

    # ---- exec / transfer ---------------------------------------------------
    def _ssh_argv(self) -> list[str]:
        host, port = self._ssh_endpoint()
        return ["ssh", "-i", self._ssh_key, *SSH_OPTS, "-p", str(port),
                f"{self.ssh_user}@{host}"]

    def _ssh_prefix(self) -> str:
        return " ".join(shlex.quote(a) for a in self._ssh_argv())

    async def _ssh_raw(self, cmd: str, timeout: float = 600) -> ExecResult:
        """Run a single command over ssh (no readiness gating)."""
        proc = await asyncio.create_subprocess_exec(
            *self._ssh_argv(), cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await _communicate(proc, timeout=timeout)
        return ExecResult(proc.returncode or 0, out, err)

    async def exec(self, cmd: str, env: dict[str, str] | None = None,
                   timeout: float | None = None) -> ExecResult:
        """Run command(s) on the box.

        No client-side timeout by default: a long training job runs until the
        box's own TTL (where the provider has one) kills it, and a *dead*
        connection is caught by ssh's ServerAlive keepalive rather than a
        wall-clock guess. Pass a finite ``timeout`` (seconds) to cap this one
        command — it raises :class:`ExecTimeoutError` on expiry (the remote
        process may keep running on the box).

        Env vars are exported *inside* the script (a fresh sshd session does not
        inherit the box's boot env), and the whole script is fed over stdin to
        ``bash -ls`` so secret values never appear in the box's argv.
        """
        exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in (env or {}).items())
        script = f"set -o pipefail\n{exports}\n{cmd}\n"
        proc = await asyncio.create_subprocess_exec(
            *self._ssh_argv(), "bash", "-ls",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await _communicate(proc, stdin=script.encode(), timeout=timeout)
        except asyncio.TimeoutError:
            head = cmd.strip().splitlines()[0][:120] if cmd.strip() else cmd
            raise ExecTimeoutError(
                f"exec timed out after {timeout:.0f}s on {self._noun} {self.id}: {head}") from None
        return ExecResult(proc.returncode or 0, out, err)

    async def push(self, local: str | Path, remote: str) -> None:
        """Upload a local directory to ``remote`` on the box (tar-over-ssh)."""
        local = str(local)
        if not Path(local).is_dir():
            raise PreflightError(f"push source not a directory: {local}")
        excl = " ".join(TAR_EXCLUDES)
        remote_cmd = f"mkdir -p {shlex.quote(remote)} && tar xzf - -C {shlex.quote(remote)}"
        pipeline = (
            f"tar czf - -C {shlex.quote(local)} {excl} . "
            f"| {self._ssh_prefix()} {shlex.quote(remote_cmd)}"
        )
        await _run_shell(pipeline, what="push")

    async def pull(self, remote: str, local_dest: str | Path) -> None:
        """Download remote dir into ``local_dest`` (creates local_dest/<basename>)."""
        local_dest = str(local_dest)
        Path(local_dest).mkdir(parents=True, exist_ok=True)
        parent = os.path.dirname(remote.rstrip("/")) or "/"
        base = os.path.basename(remote.rstrip("/"))
        remote_cmd = f"tar czf - -C {shlex.quote(parent)} {shlex.quote(base)}"
        pipeline = (
            f"{self._ssh_prefix()} {shlex.quote(remote_cmd)} "
            f"| tar xzf - -C {shlex.quote(local_dest)}"
        )
        await _run_shell(pipeline, what="pull")

    async def exists_remote(self, path: str) -> bool:
        res = await self._ssh_raw(f"test -e {shlex.quote(path)}")
        return res.exit_code == 0

    async def call(self, fn, *args, **kwargs):
        """Run a local Python function on the box; see :func:`bellhop.call.call`."""
        from .call import call as _call
        return await _call(self, fn, *args, **kwargs)

    async def exec_detached(self, cmd: str, env: dict[str, str] | None = None,
                            name: str | None = None) -> "DetachedJob":
        """Start command(s) on the box, detached from this SSH session.

        ``exec()`` runs the remote command as a child of the ssh session — if
        the *launcher* dies (laptop sleep, kill -9, network drop), sshd sends
        the job SIGHUP and a 21-hour training run dies with it. This starts
        the script under ``setsid`` with a wrapper that records the exit code,
        returns immediately, and hands back a :class:`DetachedJob` you can
        poll, tail, or ``wait()`` on — from this process or a later one
        (``DetachedJob(box, name)`` reattaches by name).

        Env vars are written into the job's script file (mode 700 dir) rather
        than argv; stdout+stderr go to ``out.log`` in the job dir.
        """
        name = name or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise PreflightError(f"detached job name must be [A-Za-z0-9._-]+, got {name!r}")
        job = DetachedJob(self, name)
        exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in (env or {}).items())
        script = f"set -o pipefail\n{exports}\n{cmd}\n"
        d = shlex.quote(job.dir)
        eof = f"BELLHOP_EOF_{uuid.uuid4().hex[:8]}"
        # exec() feeds this over stdin, so the quoted heredoc rides the same
        # channel — the script (env values included) never appears in argv.
        # The wrapper runs under setsid with all stdio detached: sshd's exit
        # can't reach it, and `echo $? > exit` is the completion signal.
        start = (
            f"mkdir -p {d} && chmod 700 {d} && cat > {d}/script.sh << '{eof}'\n"
            f"{script}{eof}\n"
            f"setsid bash -c {shlex.quote(f'bash {job.dir}/script.sh > {job.dir}/out.log 2>&1 < /dev/null; echo $? > {job.dir}/exit')} "
            f"> /dev/null 2>&1 < /dev/null & echo $! > {d}/pid"
        )
        res = await self.exec(start)
        if res.exit_code != 0:
            raise RuntimeError(
                f"exec_detached failed to start job {name!r} on {self._noun} {self.id} "
                f"(rc={res.exit_code}): {(res.stderr or res.stdout)[-500:]}")
        return job


DETACHED_DIR = "/tmp/bellhop-jobs"

# How long the watchdog's on_lifetime_expiry grace hook may run before the
# teardown proceeds anyway.
LIFETIME_GRACE_SECONDS = 900.0


class DetachedJob:
    """Handle to a detached remote job (see :meth:`SshBox.exec_detached`).

    Stateless on the client: everything lives in the job dir on the box
    (``script.sh``, ``out.log``, ``pid``, and ``exit`` once finished), so a
    fresh process can reattach with ``DetachedJob(box, name)`` and ``wait()``.
    """

    def __init__(self, box: SshBox, name: str):
        self.box = box
        self.name = name
        self.dir = f"{DETACHED_DIR}/{name}"

    async def exit_code(self) -> int | None:
        """The job's exit code, or None while it is still running."""
        res = await self.box._ssh_raw(f"cat {shlex.quote(self.dir)}/exit 2>/dev/null")
        text = res.stdout.strip()
        return int(text) if res.exit_code == 0 and text else None

    async def running(self) -> bool:
        return await self.exit_code() is None

    async def tail(self, n: int = 50) -> str:
        res = await self.box._ssh_raw(f"tail -n {int(n)} {shlex.quote(self.dir)}/out.log 2>/dev/null")
        return res.stdout

    async def wait(self, poll: float = 15.0, timeout: float | None = None) -> ExecResult:
        """Poll until the job finishes; returns its exit code + log tail.

        Like ``exec()``, unbounded by default; a finite ``timeout`` raises
        :class:`ExecTimeoutError` (the remote job keeps running — it is
        detached; that's the point).
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            code = await self.exit_code()
            if code is not None:
                return ExecResult(code, await self.tail(200), "")
            if deadline is not None and time.monotonic() >= deadline:
                raise ExecTimeoutError(
                    f"detached job {self.name!r} still running after {timeout:.0f}s "
                    f"on {self.box._noun} {self.box.id} (it keeps running; "
                    "reattach with DetachedJob(box, name))")
            await asyncio.sleep(poll)


async def _communicate(proc, stdin: bytes | None = None, timeout: float = 600):
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


async def _run_shell(pipeline: str, what: str) -> None:
    proc = await asyncio.create_subprocess_shell(
        pipeline, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed (rc={proc.returncode}): {err.decode('utf-8','replace')[:500]}")


async def wait_ready(box, probe, timeout: timedelta, interval: float) -> None:
    """Poll ``probe(box)`` until it passes; a raising probe = not ready yet.

    The shared readiness loop behind Pod._wait_ready and the Lambda/Nebius
    equivalents (each backend keeps its own *provision* wait — provider states
    differ — but "functional" is probed the same way everywhere).
    """
    deadline = time.monotonic() + timeout.total_seconds()
    while True:
        try:
            ok = await probe(box)
        except Exception:
            ok = False  # a raising probe = not ready yet (see probes.py)
        if ok:
            return
        if time.monotonic() >= deadline:
            raise PodNotReadyError(
                f"{box._noun} {box.id} provisioned but readiness probe never passed "
                f"within {timeout.total_seconds():.0f}s"
            )
        await asyncio.sleep(interval)


async def ensure_workspace(box) -> None:
    """Make ``/workspace`` exist and belong to the ssh user.

    ``run()`` lays jobs out under ``/workspace/<slug>`` on every backend.
    RunPod and Modal run as root so that just works; the ubuntu-style users on
    Lambda/Nebius can't write ``/``, but do have passwordless sudo — so this
    one-time bootstrap keeps the path contract identical across providers.
    """
    r = await box.exec('sudo -n mkdir -p /workspace && sudo -n chown "$(id -un):$(id -gn)" /workspace')
    if r.exit_code != 0:
        raise ProvisionError(
            f"could not prepare /workspace on {box._noun} {box.id} "
            f"(rc={r.exit_code}): {(r.stderr or r.stdout)[-500:]}"
        )


async def install_pip(box, specs: list[str]) -> None:
    """``config.pip`` deps-on-enter, shared by the SSH backends.

    PIP_BREAK_SYSTEM_PACKAGES: Ubuntu 24.04 (the Nebius default images) marks
    the system interpreter externally-managed (PEP 668) and plain ``pip
    install`` refuses; the env var opts out (pip>=23), and older pips on the
    RunPod/Lambda images ignore it. A disposable box has no system to protect.
    """
    quoted = " ".join(shlex.quote(s) for s in specs)
    r = await box.exec(f"python3 -m pip install -q {quoted}",
                       env={"PIP_BREAK_SYSTEM_PACKAGES": "1"})
    if r.exit_code != 0:
        raise ProvisionError(
            f"config.pip install failed on {box._noun} {box.id} "
            f"(rc={r.exit_code}): {(r.stderr or r.stdout)[-500:]}"
        )


# The -t<epoch> name stamp shared by the TTL-less backends: how the gc
# reapers know a box is bellhop's and how old it is. Names are forced to the
# bellhop- prefix at stamp time, so every launch is reapable; anything not
# matching this (hand-made boxes included) is never touched by gc.
NAME_STAMP = re.compile(r"^bellhop.*-t(\d{9,12})$")


def stamp_name(name: str) -> str:
    prefix = name if name.startswith("bellhop") else f"bellhop-{name}"
    return f"{prefix[:48]}-t{int(time.time())}"


def stamp_epoch(name: str | None) -> int | None:
    m = NAME_STAMP.match(name or "")
    return int(m.group(1)) if m else None


async def lifetime_watchdog(box, lifetime: timedelta) -> None:
    """Client-side ``max_lifetime`` enforcement for providers with no server TTL.

    Tears the box down out from under any still-running exec — its ssh
    sessions die and the exec fails, which is the intended failure mode (the
    ``cluster.py`` watchdog behaves the same way). ``_lifetime_expired`` is
    set first so the provisioning context manager can turn the resulting
    mid-run failure into a clear "hit max_lifetime" error instead of a
    baffling ssh exit-255. Only protects against forgotten boxes while *this
    process lives*; it is not a substitute for a server-side timer, which
    Lambda and Nebius simply do not offer.
    """
    await asyncio.sleep(lifetime.total_seconds())
    box._lifetime_expired = True
    print(f"bellhop: {box._noun} {box.id} hit max_lifetime {lifetime} — tearing down",
          file=sys.stderr, flush=True)
    # Grace hook: salvage what the box holds before destroying it. run()
    # points this at an emergency results pull — losing a run to its own
    # safety timer DURING the results phase is the worst possible trade.
    # Bounded so a wedged pull can't defeat the TTL entirely.
    grace = getattr(box, "on_lifetime_expiry", None)
    if grace is not None:
        try:
            await asyncio.wait_for(grace(), LIFETIME_GRACE_SECONDS)
        except Exception as e:
            print(f"bellhop: max_lifetime grace hook failed on {box._noun} {box.id}: {e}",
                  file=sys.stderr, flush=True)
    for attempt in range(3):
        try:
            await box.teardown()
            return
        except Exception as e:
            err = e
        await asyncio.sleep(10 * (attempt + 1))
    # Never bare-suppress a failed reap on a TTL-less provider: say what leaked.
    print(f"bellhop: watchdog could not tear down {box._noun} {box.id} ({err}) — "
          f"it is still billing; reap it manually", file=sys.stderr, flush=True)
