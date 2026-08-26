"""Offline tests for call() — remote function execution over the ExecBox seam.

LocalBox is a real ExecBox that "executes remotely" via local subprocesses and
directory copies, so these tests exercise the genuine pickle -> push -> runner
-> pull -> unpickle round trip (with python=sys.executable, so interpreter
parity trivially holds), without any live pod or sandbox.
"""

import asyncio
import importlib
import os
import shutil
import sys

import pytest

from bellhop import (
    PodConfig,
    PreflightError,
    ProvisionError,
    RemoteCallError,
    call,
)
from bellhop.backend import ExecResult


class LocalBox:
    """ExecBox whose 'remote' filesystem is the local one."""

    id = "local-box"

    async def exec(self, cmd, env=None, timeout=None):
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", f"set -o pipefail\n{cmd}\n",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return ExecResult(proc.returncode or 0,
                          out.decode("utf-8", "replace"), err.decode("utf-8", "replace"))

    async def push(self, local, remote):
        shutil.copytree(str(local), remote, dirs_exist_ok=True)

    async def pull(self, remote, local_dest):
        base = os.path.basename(remote.rstrip("/"))
        shutil.copytree(remote, os.path.join(str(local_dest), base), dirs_exist_ok=True)

    async def exists_remote(self, path):
        return os.path.exists(path)

    async def teardown(self):
        pass


def _call(fn, *args, **kwargs):
    # route through the local interpreter (the venv python has cloudpickle;
    # a bare `python3` might not)
    return asyncio.run(call(LocalBox(), fn, *args, python=sys.executable, **kwargs))


def test_round_trip_result():
    def add_stats(xs, scale=1.0):
        return {"n": len(xs), "sum": sum(xs) * scale}

    assert _call(add_stats, [1, 2, 3], scale=2.0) == {"n": 3, "sum": 12.0}


def test_closure_capture():
    factor = 7

    def mul(x):
        return x * factor          # closes over a local

    assert _call(mul, 6) == 42


def test_async_fn_supported():
    async def aget(x):
        await asyncio.sleep(0)
        return x + 1

    assert _call(aget, 41) == 42


def test_remote_exception_reraised_with_remote_traceback():
    def boom():
        raise ValueError("kaboom from the box")

    with pytest.raises(ValueError, match="kaboom") as ei:
        _call(boom)
    cause = ei.value.__cause__
    assert isinstance(cause, RemoteCallError)
    assert "kaboom from the box" in cause.remote_traceback
    assert "boom" in cause.remote_traceback     # remote frames, not local ones


def test_unpicklable_result_is_remote_call_error():
    def make_lock():
        import threading
        return threading.Lock()

    with pytest.raises(RemoteCallError, match="could not be pickled"):
        _call(make_lock)


def test_stdout_echoed_locally(capfd):
    def chatty():
        print("hello from the box")
        return 1

    assert _call(chatty) == 1
    assert "hello from the box" in capfd.readouterr().out


def test_remote_job_dir_cleaned_up(tmp_path):
    workdir = str(tmp_path / "remote-jobs")
    asyncio.run(call(LocalBox(), lambda: 1, python=sys.executable, workdir=workdir))
    assert os.listdir(workdir) == []            # per-call dir removed


def test_python_version_parity_mismatch_raises():
    class WrongVersionBox(LocalBox):
        async def exec(self, cmd, env=None, timeout=None):
            return ExecResult(0, "BELLHOP_PYVER=2.7\n", "")

    with pytest.raises(PreflightError, match="version mismatch"):
        asyncio.run(call(WrongVersionBox(), lambda: 1))


def test_missing_interpreter_fails_preflight():
    with pytest.raises(PreflightError, match="pre-flight failed"):
        asyncio.run(call(LocalBox(), lambda: 1, python="/nonexistent/python"))


def test_env_check_cached_per_box_and_interpreter():
    box = LocalBox()
    execs = []
    orig = box.exec

    async def counting_exec(cmd, env=None, timeout=None):
        execs.append(cmd)
        return await orig(cmd, env=env, timeout=timeout)

    box.exec = counting_exec
    asyncio.run(call(box, lambda: 1, python=sys.executable))
    n_first = len([c for c in execs if "BELLHOP_PYVER" in c])
    asyncio.run(call(box, lambda: 2, python=sys.executable))
    n_second = len([c for c in execs if "BELLHOP_PYVER" in c])
    assert n_first == 1 and n_second == 1       # pre-flight ran exactly once


def test_boxes_expose_call_method():
    from bellhop.modal_box import Sandbox
    from bellhop.pod import Pod

    for cls in (Pod, Sandbox):
        assert callable(getattr(cls, "call"))


# --- PodConfig.pip: deps-on-enter --------------------------------------------

class _OkRest:
    def __init__(self, api_key=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def create_pod(self, body):
        return {"id": "pod-1"}

    async def delete_pod(self, pod_id):
        pass


def _tmp_ssh_key(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    return str(key)


def _fake_pod_lifecycle(monkeypatch, exec_impl):
    podmod = importlib.import_module("bellhop.pod")
    monkeypatch.setattr(podmod, "RunpodRest", _OkRest)

    async def noop(self):
        return None

    monkeypatch.setattr(podmod.Pod, "_wait_provision", noop)
    monkeypatch.setattr(podmod.Pod, "_wait_ready", noop)
    monkeypatch.setattr(podmod.Pod, "exec", exec_impl)
    return podmod


def test_pod_pip_installed_on_enter(tmp_path, monkeypatch):
    calls = []

    async def fake_exec(self, cmd, env=None, timeout=None):
        calls.append(cmd)
        return ExecResult(0, "", "")

    podmod = _fake_pod_lifecycle(monkeypatch, fake_exec)
    cfg = PodConfig(ssh_key=_tmp_ssh_key(tmp_path), pip=["numpy==1.26", "tqdm"],
                    stop_after=None, terminate_after=None)

    async def _go():
        async with podmod.pod(cfg):
            pass

    asyncio.run(_go())
    (install,) = calls
    assert "pip install" in install and "numpy==1.26" in install and "tqdm" in install


def test_pod_pip_failure_raises_provision_error(tmp_path, monkeypatch):
    async def fail_exec(self, cmd, env=None, timeout=None):
        return ExecResult(1, "", "no matching distribution")

    podmod = _fake_pod_lifecycle(monkeypatch, fail_exec)
    cfg = PodConfig(ssh_key=_tmp_ssh_key(tmp_path), pip=["definitely-not-a-package"],
                    stop_after=None, terminate_after=None)

    async def _go():
        async with podmod.pod(cfg):
            pass

    with pytest.raises(ProvisionError, match="config.pip install failed"):
        asyncio.run(_go())


def test_pod_no_pip_no_extra_exec(tmp_path, monkeypatch):
    calls = []

    async def fake_exec(self, cmd, env=None, timeout=None):
        calls.append(cmd)
        return ExecResult(0, "", "")

    podmod = _fake_pod_lifecycle(monkeypatch, fake_exec)
    cfg = PodConfig(ssh_key=_tmp_ssh_key(tmp_path), stop_after=None, terminate_after=None)

    async def _go():
        async with podmod.pod(cfg):
            pass

    asyncio.run(_go())
    assert calls == []
