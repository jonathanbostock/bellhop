"""The backend seam: the provider-agnostic ephemeral-box contract.

Bellhop's identity is the imperative *check in -> run -> check out* lifecycle:
provision a disposable box, carry your code up (``push``), run steps against it
(``exec``), bring results back (``pull``), and check out (``teardown``). Four
providers implement that contract:

- **RunPod** — an SSH-able GPU/CPU pod (:class:`bellhop.pod.Pod`).
- **Modal** — an ephemeral Sandbox container (:class:`bellhop.modal_box.Sandbox`).
- **Lambda Cloud** — an on-demand GPU VM (:class:`bellhop.lambda_box.LambdaInstance`).
- **Nebius** — a GPU/CPU VM (:class:`bellhop.nebius_box.NebiusVm`).

``run()`` / ``run_many()`` (see run.py) are written against this protocol and
pick a backend purely from the config type you hand them (``PodConfig`` ->
RunPod, ``ModalConfig`` -> Modal, ``LambdaConfig`` -> Lambda, ``NebiusConfig``
-> Nebius), via :func:`open_box`.

:func:`bellhop.call.call` (remote function execution) is *derived* from these
primitives, not part of the protocol — it works over any ExecBox.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Protocol, runtime_checkable

from .errors import PreflightError

# What push() leaves out of the codebase archive, on every backend. The
# .env* patterns are a credential firewall, not tidiness: a repo-root .env
# full of API keys must never land on a (possibly community-hosted) box.
# Ship box-side secrets explicitly via exec(env=...) / RunSpec(env=...),
# which never touch the box's disk or argv.
TAR_EXCLUDES = ["--exclude=.git", "--exclude=__pycache__", "--exclude=.venv",
                "--exclude=node_modules", "--exclude=*.pyc",
                "--exclude=.env", "--exclude=.env.*"]


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


@runtime_checkable
class ExecBox(Protocol):
    """A live, execable ephemeral box. Pod, Sandbox, LambdaInstance and
    NebiusVm all satisfy this."""

    id: str

    # timeout: optional *client-side* cap in seconds; None (default) means no
    # client-side limit — the box's TTL is the backstop (server-side on
    # RunPod/Modal; the in-process max_lifetime watchdog on Lambda/Nebius).
    # A finite timeout raises ExecTimeoutError when it expires.
    async def exec(self, cmd: str, env: dict[str, str] | None = None,
                   timeout: float | None = None) -> ExecResult: ...

    async def push(self, local: str | Path, remote: str) -> None: ...

    async def pull(self, remote: str, local_dest: str | Path) -> None: ...

    async def exists_remote(self, path: str) -> bool: ...

    async def teardown(self) -> None: ...


@contextlib.asynccontextmanager
async def open_box(backend, *, keep: bool = False,
                   api_key: str | None = None) -> AsyncIterator[ExecBox]:
    """Provision the box implied by ``backend``'s type, yield it, tear it down.

    Dispatches on the config class so callers never branch on provider:
    ``PodConfig`` -> RunPod pod, ``ModalConfig`` -> Modal sandbox,
    ``LambdaConfig`` -> Lambda instance, ``NebiusConfig`` -> Nebius VM.
    Imports are local so a RunPod-only install never needs ``modal`` or
    ``nebius`` (and vice versa). ``api_key`` is the *RunPod* key — the other
    providers use their own ambient auth (LAMBDA_API_KEY, Modal token,
    Nebius IAM credentials).
    """
    # Local imports avoid a circular dependency (pod/modal_box import this
    # module for ExecResult) and keep provider deps optional.
    from .pod import PodConfig, pod

    if isinstance(backend, PodConfig):
        async with pod(backend, keep=keep, api_key=api_key) as p:
            yield p
        return

    from .lambda_box import LambdaConfig, instance

    if isinstance(backend, LambdaConfig):
        async with instance(backend, keep=keep) as i:
            yield i
        return

    from .modal_box import ModalConfig, sandbox

    if isinstance(backend, ModalConfig):
        async with sandbox(backend, keep=keep) as s:
            yield s
        return

    from .nebius_box import NebiusConfig, vm

    if isinstance(backend, NebiusConfig):
        async with vm(backend, keep=keep) as v:
            yield v
        return

    raise PreflightError(
        f"unknown backend config {type(backend).__name__!r}; expected PodConfig "
        "(RunPod), ModalConfig (Modal), LambdaConfig (Lambda) or NebiusConfig (Nebius)"
    )
