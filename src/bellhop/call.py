"""Remote function execution — Modal ergonomics on any ExecBox.

``call(box, fn, *args, **kwargs)`` runs a plain local Python function on a live
box and returns its result as a Python object — no script file, no CLI
entrypoint, no stdout parsing. It is layered entirely on the ExecBox
primitives (push / exec / pull), so it works identically on a RunPod ``Pod``
and a Modal ``Sandbox``; both expose it as a ``.call()`` method.

Mechanics: the function + arguments are serialized with ``cloudpickle``,
pushed to the box, executed by a small runner script under the box's Python,
and the pickled result (or exception) is pulled back. A remote exception is
re-raised locally as its original type, chained to a
:class:`~bellhop.errors.RemoteCallError` carrying the remote traceback.

The one hard prerequisite is **interpreter parity**: cloudpickle serializes
code objects, which are not portable across Python *minor* versions (a 3.12
devbox cannot call into a 3.11 image). The first ``call()`` on a box
pre-flights this and fails fast with a clear message instead of letting the
mismatch surface as a corrupt-pickle error mid-job. ``cloudpickle`` itself is
auto-installed on the box (pinned to the local version) if missing.

Reserved keyword arguments (``timeout``, ``python``, ``echo``, ``workdir``)
are consumed by ``call()`` itself; a function that needs a parameter with one
of those names should be wrapped with ``functools.partial`` first.
"""

from __future__ import annotations

import contextlib
import shlex
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cloudpickle

from .errors import PreflightError, RemoteCallError, RemoteJobError

DEFAULT_WORKDIR = "/tmp/bellhop-call"

# Runs on the box with cwd = the pushed job dir. Exit 0 whenever the runner
# itself worked — a raising *user* function is a result (recorded in
# out/result.pkl), not an infra failure; non-zero exit means the harness broke.
_RUNNER = '''\
import os, sys, traceback
import cloudpickle

def main():
    with open("payload.pkl", "rb") as f:
        fn, args, kwargs = cloudpickle.load(f)
    out = {"ok": True, "value": None, "exc": None, "tb": None}
    try:
        import inspect
        if inspect.iscoroutinefunction(fn):
            import asyncio
            out["value"] = asyncio.run(fn(*args, **kwargs))
        else:
            out["value"] = fn(*args, **kwargs)
    except BaseException:
        out.update(ok=False, value=None, tb=traceback.format_exc())
        try:
            out["exc"] = cloudpickle.dumps(sys.exc_info()[1])
        except Exception:
            out["exc"] = None
    try:
        blob = cloudpickle.dumps(out)
    except Exception:
        out = {"ok": False, "value": None, "exc": None,
               "tb": "return value could not be pickled:\\n" + traceback.format_exc()}
        blob = cloudpickle.dumps(out)
    os.makedirs("out", exist_ok=True)
    with open("out/result.pkl", "wb") as f:
        f.write(blob)

main()
'''


async def _ensure_call_env(box, python: str) -> None:
    """One-time per (box, interpreter): check version parity, ensure cloudpickle.

    Runs lazily on the first call() so plain exec/push/pull users pay nothing.
    """
    ready: set = getattr(box, "_bellhop_call_envs", None) or set()
    if python in ready:
        return
    q = shlex.quote(python)
    pin = shlex.quote(f"cloudpickle=={cloudpickle.__version__}")
    script = (
        f"{q} -c 'import sys; print(\"BELLHOP_PYVER=%d.%d\" % sys.version_info[:2])' && "
        f"({q} -c 'import cloudpickle' 2>/dev/null "
        f"|| {q} -m pip install -q {pin} || {q} -m pip install -q cloudpickle)"
    )
    res = await box.exec(script)
    if res.exit_code != 0:
        raise PreflightError(
            f"call() pre-flight failed on box {box.id} (rc={res.exit_code}): "
            f"could not run {python!r} / install cloudpickle — "
            f"{(res.stderr or res.stdout)[-500:]}"
        )
    remote = next((ln.split("=", 1)[1].strip() for ln in res.stdout.splitlines()
                   if ln.startswith("BELLHOP_PYVER=")), None)
    local = "%d.%d" % sys.version_info[:2]
    if remote != local:
        raise PreflightError(
            f"Python version mismatch: local {local} vs {remote} on box {box.id} "
            f"({python}). cloudpickle'd code objects are not portable across "
            "minor versions — pick an image with a matching Python, or pass "
            "python= pointing at a matching interpreter on the box."
        )
    ready.add(python)
    box._bellhop_call_envs = ready


async def call(box, fn, *args: Any, timeout: float | None = None,
               python: str = "python3", echo: bool = True,
               workdir: str = DEFAULT_WORKDIR, **kwargs: Any) -> Any:
    """Run ``fn(*args, **kwargs)`` on the box and return its result.

    - ``fn`` may be sync or async (async is driven with ``asyncio.run`` on the
      box). Its stdout/stderr are relayed locally after completion when
      ``echo=True`` (output is buffered by exec, not streamed).
    - ``timeout`` caps this one call client-side (seconds), raising
      :class:`~bellhop.errors.ExecTimeoutError` like any exec.
    - A remote exception is re-raised locally as its original type, with a
      :class:`~bellhop.errors.RemoteCallError` (carrying the remote traceback)
      as its ``__cause__``; if the exception object itself can't be pickled,
      the RemoteCallError is raised directly.
    - Arguments and results travel by value (cloudpickle): fine for configs
      and metrics; large artifacts belong on GCS / a volume, with paths in the
      payload.
    """
    await _ensure_call_env(box, python)
    job_dir = f"{workdir.rstrip('/')}/{uuid.uuid4().hex[:12]}"
    try:
        with tempfile.TemporaryDirectory() as td:
            Path(td, "payload.pkl").write_bytes(cloudpickle.dumps((fn, args, kwargs)))
            Path(td, "_bellhop_runner.py").write_text(_RUNNER)
            await box.push(td, job_dir)
            res = await box.exec(
                f"cd {shlex.quote(job_dir)} && {shlex.quote(python)} _bellhop_runner.py",
                timeout=timeout,
            )
            if echo:
                if res.stdout:
                    sys.stdout.write(res.stdout)
                if res.stderr:
                    sys.stderr.write(res.stderr)
            if res.exit_code != 0:
                raise RemoteJobError(
                    f"call() runner crashed on box {box.id} (rc={res.exit_code}): "
                    f"{(res.stderr or res.stdout)[-500:]}",
                    remote_exit=res.exit_code, log_tail=res.stderr[-2000:],
                )
            ret_dir = Path(td, "ret")
            await box.pull(f"{job_dir}/out", ret_dir)
            result_file = ret_dir / "out" / "result.pkl"
            if not result_file.exists():
                raise RemoteJobError(
                    f"call() runner exited 0 on box {box.id} but produced no result.pkl",
                    remote_exit=0,
                )
            out = cloudpickle.loads(result_file.read_bytes())
    finally:
        with contextlib.suppress(Exception):
            await box.exec(f"rm -rf {shlex.quote(job_dir)}")

    if out["ok"]:
        return out["value"]
    err = RemoteCallError(
        f"remote function raised on box {box.id} — remote traceback:\n{out['tb']}",
        remote_traceback=out["tb"] or "",
    )
    exc = None
    if out.get("exc"):
        with contextlib.suppress(Exception):
            exc = cloudpickle.loads(out["exc"])
    if isinstance(exc, BaseException):
        raise exc from err
    raise err
