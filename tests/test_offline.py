"""Offline unit tests — pure logic, no live pod / sandbox."""

import os
from datetime import timedelta

import pytest

from bellhop import (
    BellhopError,
    ModalConfig,
    PodConfig,
    PreflightError,
    ProvisionError,
    RemoteJobError,
    RunpodError,
    open_box,
)
from bellhop.modal_box import _create_kwargs
from bellhop.run import _is_git


def _cfg(**kw):
    # supply a fake key pair so pubkey/key resolution doesn't touch the real ~/.ssh
    return PodConfig(**kw)


def test_image_resolution_default_gpu():
    assert "pytorch" in _cfg(compute="gpu", gpu_id="x").resolve_image()


def test_image_resolution_preset_and_freeform():
    assert _cfg(image_preset="cpu-base", compute="cpu").resolve_image() == "runpod/base:1.0.2-ubuntu2204"
    assert _cfg(image="my/custom:1", image_preset="cpu-base").resolve_image() == "my/custom:1"


def test_unknown_preset_raises():
    with pytest.raises(PreflightError):
        _cfg(image_preset="does-not-exist").resolve_image()


def test_gpu_requires_gpu_id(tmp_path, monkeypatch):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    cfg = _cfg(compute="gpu", gpu_id=None, ssh_key=str(key))
    with pytest.raises(PreflightError):
        cfg.to_create_body()


# --- unified gpu= vocabulary -------------------------------------------------

def test_gpu_alias_expands_to_candidate_list():
    assert _cfg(gpu="A100").resolve_gpu_ids() == [
        "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
    # normalization: case / dashes / spaces don't matter
    assert _cfg(gpu="a100").resolve_gpu_ids() == _cfg(gpu="A-100").resolve_gpu_ids()
    assert _cfg(gpu="rtx 4090").resolve_gpu_ids() == ["NVIDIA GeForce RTX 4090"]


def test_gpu_full_runpod_id_passes_verbatim():
    assert _cfg(gpu="NVIDIA GeForce RTX 4090").resolve_gpu_ids() == ["NVIDIA GeForce RTX 4090"]


def test_gpu_unknown_short_name_raises():
    with pytest.raises(PreflightError, match="known aliases"):
        _cfg(gpu="Z9000").resolve_gpu_ids()


def test_gpu_and_gpu_id_both_set_raises():
    with pytest.raises(PreflightError, match="not both"):
        _cfg(gpu="A100", gpu_id="NVIDIA A100 80GB PCIe").resolve_gpu_ids()


def test_compute_derived_from_gpu():
    assert _cfg().resolved_compute == "cpu"                    # no gpu -> CPU box
    assert _cfg(gpu="A100").resolved_compute == "gpu"
    assert _cfg(gpu_id="NVIDIA L4").resolved_compute == "gpu"
    assert _cfg(compute="cpu", ).resolved_compute == "cpu"     # explicit wins
    assert _cfg(compute="gpu", gpu_id="x").resolved_compute == "gpu"


def test_create_body_uses_alias_candidates(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    body = _cfg(gpu="H100", ssh_key=str(key)).to_create_body()
    assert body["gpuTypeIds"] == [
        "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"]


def test_graphql_input_takes_candidate_override(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    cfg = _cfg(gpu="A100", ssh_key=str(key))
    assert cfg.to_graphql_input()["gpuTypeId"] == "NVIDIA A100 80GB PCIe"
    assert cfg.to_graphql_input(gpu_type_id="NVIDIA A100-SXM4-80GB")["gpuTypeId"] == "NVIDIA A100-SXM4-80GB"


# --- unified max_lifetime= ---------------------------------------------------

def test_max_lifetime_maps_to_terminate_after():
    cfg = _cfg(gpu="A100", max_lifetime=timedelta(hours=8))
    assert cfg.terminate_after == timedelta(hours=8)
    # survives dataclasses.replace (run() re-names the config per slug)
    from dataclasses import replace
    assert replace(cfg, name="x").terminate_after == timedelta(hours=8)


def test_max_lifetime_clears_stop_after():
    # The default 24h stop_after would halt a 48h job at the day mark even
    # though the user asked for max_lifetime=48h — max_lifetime wins over both timers.
    from dataclasses import replace
    cfg = _cfg(gpu="A100", max_lifetime=timedelta(hours=48))
    assert cfg.terminate_after == timedelta(hours=48)
    assert cfg.stop_after is None
    assert replace(cfg, name="x").stop_after is None


def test_readiness_timeout_defaults():
    cfg = _cfg(gpu="A100")
    assert cfg.provision_timeout == timedelta(seconds=300)
    assert cfg.ready_timeout == timedelta(seconds=420)


def test_readiness_timeouts_widen_for_docker_start_cmd():
    # a bootstrap that apt-installs sshd routinely needs 5-15 min (issue #27)
    cfg = _cfg(gpu="A100", docker_start_cmd="apt-get install -y openssh-server && /usr/sbin/sshd -D")
    assert cfg.provision_timeout == timedelta(seconds=1200)
    assert cfg.ready_timeout == timedelta(seconds=1200)


def test_readiness_timeout_explicit_wins_over_docker_start_cmd():
    from dataclasses import replace
    cfg = _cfg(gpu="A100", docker_start_cmd="sleep infinity",
               provision_timeout=timedelta(seconds=60), ready_timeout=timedelta(seconds=90))
    assert cfg.provision_timeout == timedelta(seconds=60)
    assert cfg.ready_timeout == timedelta(seconds=90)
    # resolved values are concrete, so replace() keeps them
    assert replace(cfg, name="x").ready_timeout == timedelta(seconds=90)


def test_max_lifetime_maps_to_modal_timeout():
    kw = _create_kwargs(ModalConfig(max_lifetime=timedelta(hours=8)))
    assert kw["timeout"] == 8 * 3600


def test_create_body_shape(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    body = _cfg(compute="gpu", gpu_id="NVIDIA GeForce RTX 4090", ssh_key=str(key)).to_create_body()
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert body["env"]["PUBLIC_KEY"].startswith("ssh-ed25519")
    assert body["ports"] == ["22/tcp"]
    assert body["cloudType"] == "COMMUNITY"


def test_missing_ssh_key_raises():
    with pytest.raises(PreflightError):
        _cfg(compute="gpu", gpu_id="x", ssh_key="/nope/missing").to_create_body()


def test_is_git_detection():
    assert _is_git("https://github.com/x/y")
    assert _is_git("git@github.com:x/y.git")
    assert not _is_git("./local/dir")


def test_error_exit_codes():
    assert PreflightError.exit_code == 10
    assert ProvisionError.exit_code == 20
    assert RemoteJobError("x", remote_exit=7).exit_code == 40
    assert RemoteJobError("x", remote_exit=7).remote_exit == 7
    assert issubclass(PreflightError, RunpodError)


def test_graphql_ttl_input(tmp_path):
    from datetime import timedelta
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    cfg = _cfg(compute="gpu", gpu_id="NVIDIA GeForce RTX 4090", ssh_key=str(key),
               stop_after=timedelta(hours=24), terminate_after=timedelta(hours=72))
    assert cfg.has_ttl()
    gi = cfg.to_graphql_input()
    # GraphQL shape differs from REST: singular gpuTypeId, ports string, env list
    assert gi["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert gi["ports"] == "22/tcp"
    assert {"key": "PUBLIC_KEY", "value": "ssh-ed25519 AAAA test"} in gi["env"]
    assert gi["stopAfter"].endswith("Z") and gi["terminateAfter"].endswith("Z")
    assert gi["stopAfter"] < gi["terminateAfter"]


def test_ttl_disabled_when_none(tmp_path):
    cfg = _cfg(compute="gpu", gpu_id="x", stop_after=None, terminate_after=None)
    assert not cfg.has_ttl()


def test_graphql_ttl_requires_gpu(tmp_path):
    from datetime import timedelta
    cfg = _cfg(compute="cpu", stop_after=timedelta(hours=1))
    with pytest.raises(PreflightError):
        cfg.to_graphql_input()


def test_cpu_with_ttl_still_builds_rest_body(tmp_path):
    # CPU + default TTL must NOT crash: it falls back to the REST path, which
    # has no native timer. Guards the routing fix in pod().
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    cfg = _cfg(compute="cpu", ssh_key=str(key))  # has_ttl() True by default
    assert cfg.has_ttl()
    body = cfg.to_create_body()
    assert body["computeType"] == "CPU"


def test_missing_api_key_raises(monkeypatch):
    from bellhop.rest import _api_key
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(PreflightError):
        _api_key(None)


# --- Modal backend (no network / no real sandbox) ---------------------------

def test_runpod_error_is_bellhop_alias():
    assert RunpodError is BellhopError
    assert issubclass(ProvisionError, BellhopError)


def test_modal_create_kwargs_gpu_and_ttl():
    cfg = ModalConfig(gpu="A100", cpu=2.0, memory=4096,
                      timeout=timedelta(hours=2), idle_timeout=timedelta(minutes=10))
    kw = _create_kwargs(cfg)
    assert kw["gpu"] == "A100"
    assert kw["cpu"] == 2.0 and kw["memory"] == 4096
    assert kw["timeout"] == 7200          # hard max lifetime, seconds
    assert kw["idle_timeout"] == 600      # idle terminate, seconds


def test_modal_create_kwargs_defaults():
    kw = _create_kwargs(ModalConfig())    # CPU box, default TTL
    assert "gpu" not in kw                # None -> omitted (CPU)
    assert kw["timeout"] == 24 * 3600     # default 24h
    assert "idle_timeout" not in kw       # None -> omitted


def test_modal_ttl_none_omits_timeout_and_warns():
    # timeout=None does NOT mean "no TTL" on Modal (Modal enforces its own
    # 300s default) — the kwarg is omitted and the user is warned loudly.
    with pytest.warns(UserWarning, match="300s"):
        kw = _create_kwargs(ModalConfig(timeout=None))
    assert "timeout" not in kw


def test_modal_env_and_volumes_only_when_set():
    assert "env" not in _create_kwargs(ModalConfig())
    assert _create_kwargs(ModalConfig(env={"K": "v"}))["env"] == {"K": "v"}


def test_open_box_rejects_unknown_backend():
    import asyncio

    async def _go():
        async with open_box(object()):
            pass

    with pytest.raises(PreflightError):
        asyncio.run(_go())


# Image resolution actually builds a modal.Image, so it needs the extra.
def test_modal_image_resolution_default_and_preset():
    pytest.importorskip("modal")
    import modal

    assert isinstance(ModalConfig().resolve_image(), modal.Image)
    assert isinstance(ModalConfig(image_preset="debian-slim").resolve_image(), modal.Image)
    # a registry string is wrapped via from_registry
    assert isinstance(ModalConfig(image="python:3.11-slim").resolve_image(), modal.Image)


def test_modal_unknown_preset_raises():
    pytest.importorskip("modal")
    with pytest.raises(PreflightError):
        ModalConfig(image_preset="does-not-exist").resolve_image()


# --- exec timeout: unbounded by default, opt-in cap (issue #4) ----------------

def test_exec_default_timeout_is_unbounded():
    import inspect

    from bellhop.backend import ExecBox
    from bellhop.modal_box import Sandbox
    from bellhop.pod import Pod

    for cls in (ExecBox, Pod, Sandbox):
        assert inspect.signature(cls.exec).parameters["timeout"].default is None


def test_exec_finite_timeout_raises_exec_timeout_error(monkeypatch):
    import asyncio

    from bellhop import ExecTimeoutError
    from bellhop.pod import Pod

    class _HangProc:
        returncode = None
        killed = False

        async def communicate(self, stdin=None):
            await asyncio.sleep(30)

        def kill(self):
            self.killed = True

        async def wait(self):
            return 0

    hang = _HangProc()

    async def fake_exec(*argv, **kw):
        return hang

    p = Pod.__new__(Pod)          # skip provisioning; exec only needs _ssh_argv
    p.id = "pod-x"
    monkeypatch.setattr(Pod, "_ssh_argv", lambda self: ["ssh", "fake"])
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    with pytest.raises(ExecTimeoutError, match="timed out after"):
        asyncio.run(p.exec("sleep 999", timeout=0.05))
    assert hang.killed            # the local ssh process was reaped


def test_runspec_timeout_plumbs_to_job_exec_only(tmp_path, monkeypatch):
    import asyncio
    import contextlib
    import importlib

    from bellhop.backend import ExecResult
    from bellhop.run import RunSpec

    runmod = importlib.import_module("bellhop.run")

    calls = []

    class FakeBox:
        id = "fake"

        async def exec(self, cmd, env=None, timeout=None):
            calls.append((cmd, timeout))
            return ExecResult(0, "", "")

        async def push(self, local, remote):
            pass

        async def pull(self, remote, dest):
            pass

        async def exists_remote(self, path):
            return True

        async def teardown(self):
            pass

    @contextlib.asynccontextmanager
    async def fake_open_box(backend, *, keep=False, api_key=None):
        yield FakeBox()

    monkeypatch.setattr(runmod, "open_box", fake_open_box)
    spec = RunSpec(slug="s", codebase=str(tmp_path), run="python x.py",
                   local_out=str(tmp_path / "out"), gcs_base=None,
                   timeout=7200)
    res = asyncio.run(runmod.run(spec, PodConfig()))
    assert res.remote_exit == 0

    job_calls = [t for cmd, t in calls if "--- run ---" in cmd]
    setup_calls = [t for cmd, t in calls if "--- run ---" not in cmd]
    assert job_calls == [7200]                      # the job gets the cap
    assert all(t is None for t in setup_calls)      # housekeeping stays unbounded


def test_runspec_timeout_defaults_to_none():
    from bellhop.run import RunSpec

    assert RunSpec(slug="s", codebase=".", run="x").timeout is None


# --- fixes from the 0.5.0 diagnosis pass --------------------------------------

def test_wait_ready_treats_probe_raise_as_not_ready():
    # probes.py promises "raising is also treated as not-ready"; a custom probe
    # that raises must be retried, not crash the provision.
    import asyncio

    from bellhop.pod import Pod

    class FlakyProbe:
        calls = 0

        async def __call__(self, pod):
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("transient")
            return True

    p = Pod.__new__(Pod)
    p.id = "pod-x"
    p.config = PodConfig(ready=FlakyProbe(), ready_timeout=timedelta(seconds=5),
                         poll_interval=0.01)
    asyncio.run(p._wait_ready())
    assert FlakyProbe.calls == 2


class _FakeRest:
    """Stands in for RunpodRest in pod() routing tests."""

    fail_with = "no capacity on {cloud}"

    def __init__(self, api_key=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def create_pod(self, body):
        raise ProvisionError(self.fail_with.format(cloud=body["cloudType"]))


def _tmp_ssh_key(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    return str(key)


def test_cloud_fallback_reports_both_errors(tmp_path, monkeypatch):
    import asyncio
    import importlib

    podmod = importlib.import_module("bellhop.pod")
    monkeypatch.setattr(podmod, "RunpodRest", _FakeRest)
    cfg = PodConfig(ssh_key=_tmp_ssh_key(tmp_path), stop_after=None, terminate_after=None)

    async def _go():
        async with podmod.pod(cfg):
            pass

    with pytest.raises(ProvisionError, match="COMMUNITY.*SECURE fallback"):
        asyncio.run(_go())


def test_gql_create_reports_every_attempt(tmp_path, monkeypatch):
    # RunPod's GraphQL errors are opaque one-liners; raising only the last
    # attempt's error hid the COMMUNITY/SECURE ladder entirely (issue #27).
    import asyncio
    import importlib

    podmod = importlib.import_module("bellhop.pod")

    class _FakeGql:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def create_pod_on_demand(self, gi):
            raise ProvisionError(
                f"graphql error: no dice for {gi['gpuTypeId']} on {gi['cloudType']}"
            )

    monkeypatch.setattr(podmod, "RunpodGraphQL", _FakeGql)
    cfg = PodConfig(gpu="H100", ssh_key=_tmp_ssh_key(tmp_path))  # default TTL -> gql path
    with pytest.raises(ProvisionError) as ei:
        asyncio.run(podmod._gql_create(cfg, api_key="x"))
    msg = str(ei.value)
    # every candidate on both clouds, not just the final attempt
    for gid in cfg.resolve_gpu_ids():
        assert f"{gid} on COMMUNITY" in msg and f"{gid} on SECURE" in msg


def test_is_capacity_error_classification():
    from bellhop import is_capacity_error

    # real prose from live runs (issue #27 probe matrix)
    assert is_capacity_error(ProvisionError(
        "graphql create failed on every cloud/GPU attempt:\n"
        "  NVIDIA GeForce RTX 4090 on COMMUNITY: graphql error: This machine "
        "does not have the resources to deploy your pod. Please try a different machine"
    ))
    assert is_capacity_error(ProvisionError(
        "podFindAndDeployOnDemand returned null (no capacity for the request)"
    ))
    # a broken request must NOT be classified as capacity
    assert not is_capacity_error(ProvisionError(
        'graphql error: Variable "$input" got invalid value'
    ))
    assert not is_capacity_error(ProvisionError("graphql HTTP 401: unauthorized"))


def test_cpu_pod_with_ttl_warns(tmp_path, monkeypatch):
    # CPU pods can't get a native TTL (GraphQL on-demand is GPU-only); the
    # silent drop was a leaked-pod risk, so it must warn.
    import asyncio
    import importlib

    podmod = importlib.import_module("bellhop.pod")
    monkeypatch.setattr(podmod, "RunpodRest", _FakeRest)
    cfg = PodConfig(ssh_key=_tmp_ssh_key(tmp_path), cloud_fallback=False)  # default TTL, CPU
    assert cfg.has_ttl()

    async def _go():
        async with podmod.pod(cfg):
            pass

    with pytest.warns(UserWarning, match="GPU-only"):
        with pytest.raises(ProvisionError):
            asyncio.run(_go())


# --- Modal exec honors the ExecBox timeout contract ---------------------------

class _Aio:
    def __init__(self, fn):
        self.aio = fn


class _FakeStream:
    def __init__(self, data=""):
        self._d = data
        self.read = _Aio(self._read)

    async def _read(self):
        return self._d


class _FakeProc:
    def __init__(self, code, delay=0.0):
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self._code = code
        self._delay = delay
        self.wait = _Aio(self._wait)

    async def _wait(self):
        import asyncio

        if self._delay:
            await asyncio.sleep(self._delay)
        return self._code


class _FakeSb:
    object_id = "sb-x"

    def __init__(self, proc):
        self._proc = proc
        self.exec = _Aio(self._exec)

    async def _exec(self, *a, **kw):
        return self._proc


def test_modal_exec_timeout_raises_exec_timeout_error():
    # Modal enforces its native timeout by killing the process (plain non-zero
    # exit) — the backend must translate that into ExecTimeoutError per ExecBox.
    import asyncio

    from bellhop import ExecTimeoutError
    from bellhop.modal_box import Sandbox

    box = Sandbox(_FakeSb(_FakeProc(code=137, delay=0.1)), ModalConfig())
    with pytest.raises(ExecTimeoutError, match="timed out after"):
        asyncio.run(box.exec("sleep 999", timeout=0.05))


def test_modal_exec_fast_failure_is_not_a_timeout():
    import asyncio

    from bellhop.modal_box import Sandbox

    box = Sandbox(_FakeSb(_FakeProc(code=1)), ModalConfig())
    res = asyncio.run(box.exec("false", timeout=60))
    assert res.exit_code == 1             # legit failure passes through


# --- run() job script + salvage behavior --------------------------------------

def test_job_script_aborts_on_setup_failure(tmp_path):
    # A failing setup must abort the job (set -e in the block), not silently
    # run against a half-configured box. Execute the actual script with bash.
    import subprocess

    from bellhop.run import RunSpec, _job_script

    spec = RunSpec(slug="s", codebase=".", setup="false",
                   run="echo SHOULD_NOT_RUN")
    script = "set -o pipefail\n" + _job_script(spec, str(tmp_path))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode != 0
    assert "SHOULD_NOT_RUN" not in r.stdout
    log = (tmp_path / "results" / "run.log").read_text()
    assert "SHOULD_NOT_RUN" not in log


def test_job_script_exit_status_is_the_jobs(tmp_path):
    import subprocess

    from bellhop.run import RunSpec, _job_script

    spec = RunSpec(slug="s", codebase=".", run="echo ok; exit 7")
    script = "set -o pipefail\n" + _job_script(spec, str(tmp_path))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 7              # tee doesn't mask the job's status
    assert "ok" in (tmp_path / "results" / "run.log").read_text()


class _SalvageBox:
    """FakeBox whose job exec times out; records whether pull still ran."""

    id = "fake"

    def __init__(self):
        self.pulled = []

    async def exec(self, cmd, env=None, timeout=None):
        from bellhop.backend import ExecResult
        from bellhop.errors import ExecTimeoutError

        if "--- run ---" in cmd:
            raise ExecTimeoutError("exec timed out after 1s")
        return ExecResult(0, "", "")

    async def push(self, local, remote):
        pass

    async def pull(self, remote, dest):
        self.pulled.append(remote)

    async def exists_remote(self, path):
        return True

    async def teardown(self):
        pass


def test_run_salvages_results_on_timeout(tmp_path, monkeypatch):
    import asyncio
    import contextlib
    import importlib

    from bellhop import ExecTimeoutError
    from bellhop.run import RunSpec

    runmod = importlib.import_module("bellhop.run")
    box = _SalvageBox()

    @contextlib.asynccontextmanager
    async def fake_open_box(backend, *, keep=False, api_key=None):
        yield box

    monkeypatch.setattr(runmod, "open_box", fake_open_box)
    spec = RunSpec(slug="s", codebase=str(tmp_path), run="python x.py",
                   local_out=str(tmp_path / "out"), gcs_base=None, timeout=1)
    with pytest.raises(ExecTimeoutError):
        asyncio.run(runmod.run(spec, PodConfig()))
    assert box.pulled == ["/workspace/s/results"]   # partials came back first


def test_nested_results_subdir_log_tail(tmp_path, monkeypatch):
    # pull() extracts to local_out/<basename(remote)>; log_tail must look there
    # even when results_subdir is nested.
    import asyncio
    import contextlib
    import importlib
    import os

    from bellhop.backend import ExecResult
    from bellhop.run import RunSpec

    runmod = importlib.import_module("bellhop.run")

    class Box:
        id = "fake"

        async def exec(self, cmd, env=None, timeout=None):
            return ExecResult(0, "", "")

        async def push(self, local, remote):
            pass

        async def pull(self, remote, dest):
            # mimic real pull: dest/<basename(remote)>/
            d = os.path.join(dest, os.path.basename(remote.rstrip("/")))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "run.log"), "w") as f:
                f.write("hello from the log\n")

        async def exists_remote(self, path):
            return True

        async def teardown(self):
            pass

    @contextlib.asynccontextmanager
    async def fake_open_box(backend, *, keep=False, api_key=None):
        yield Box()

    monkeypatch.setattr(runmod, "open_box", fake_open_box)
    spec = RunSpec(slug="s", codebase=str(tmp_path), run="x",
                   results_subdir="out/results",
                   local_out=str(tmp_path / "loc"), gcs_base=None)
    res = asyncio.run(runmod.run(spec, PodConfig()))
    assert "hello from the log" in res.log_tail
