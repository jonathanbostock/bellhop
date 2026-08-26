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
import shlex
import sys
from datetime import timedelta
from pathlib import Path

from .backend import TAR_EXCLUDES, ExecResult
from .errors import ExecTimeoutError, PreflightError

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


async def lifetime_watchdog(box, lifetime: timedelta) -> None:
    """Client-side ``max_lifetime`` enforcement for providers with no server TTL.

    Tears the box down out from under any still-running exec — its ssh
    sessions die and the exec fails, which is the intended failure mode (the
    ``cluster.py`` watchdog behaves the same way). Only protects against
    forgotten boxes while *this process lives*; it is not a substitute for a
    server-side timer, which Lambda and Nebius simply do not offer.
    """
    await asyncio.sleep(lifetime.total_seconds())
    print(f"bellhop: {box._noun} {box.id} hit max_lifetime {lifetime} — tearing down",
          file=sys.stderr, flush=True)
    with contextlib.suppress(Exception):
        await box.teardown()
