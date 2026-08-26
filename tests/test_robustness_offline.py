"""Offline tests for the robustness features driven by real training-run
casualties (see docs/design/reliable-providers.md "Field reports"): failure-
aware keeps, host floors + reroll, detached exec, .env exclusion, watchdog
grace, and failure-log persistence. No live box, no cost.
"""

import asyncio
import contextlib
from datetime import timedelta

import httpx
import pytest

from bellhop import (
    LambdaConfig,
    LambdaInstance,
    NebiusConfig,
    PodConfig,
    PreflightError,
    ProvisionError,
    RemoteJobError,
    RunSpec,
)
from bellhop.backend import TAR_EXCLUDES, ExecResult
from bellhop.lambda_box import LambdaRest
from bellhop.pod import Pod
from bellhop.sshbox import DetachedJob, SshBox, lifetime_watchdog


@pytest.fixture()
def keypair(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAAtest bellhop@test")
    return str(key)


# --- .env must never ship to a box --------------------------------------------

def test_env_files_excluded_from_push():
    assert "--exclude=.env" in TAR_EXCLUDES
    assert "--exclude=.env.*" in TAR_EXCLUDES


# --- host floors: minMemoryInGb / minVcpuCount --------------------------------

def test_host_floors_in_graphql_input(keypair):
    cfg = PodConfig(gpu="H100", ssh_key=keypair, min_memory_gb=1900, min_vcpu=96)
    gi = cfg.to_graphql_input()
    assert gi["minMemoryInGb"] == 1900
    assert gi["minVcpuCount"] == 96


def test_host_floors_route_create_through_graphql(keypair):
    # REST v1 has no floor fields — floors must force the GraphQL path even
    # with every TTL disabled.
    cfg = PodConfig(gpu="H100", ssh_key=keypair, min_memory_gb=1900,
                    stop_after=None, terminate_after=None)
    assert not cfg.has_ttl() and cfg.has_host_floor() and cfg.needs_graphql()
    assert "minMemoryInGb" not in PodConfig(gpu="H100", ssh_key=keypair).to_graphql_input()


# --- host_check: raise-to-reroll -----------------------------------------------

class _FakeRest:
    """Stands in for RunpodRest: hands out sequential pod ids, records deletes."""

    created: list = []
    deleted: list = []

    def __init__(self, api_key=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def create_pod(self, body):
        pid = f"pod-{len(_FakeRest.created)}"
        _FakeRest.created.append(pid)
        return {"id": pid}

    async def delete_pod(self, pod_id):
        _FakeRest.deleted.append(pod_id)


@pytest.fixture()
def fake_pod_backend(monkeypatch, keypair):
    import importlib

    podmod = importlib.import_module("bellhop.pod")  # bellhop.pod the ATTR is the function
    _FakeRest.created, _FakeRest.deleted = [], []
    monkeypatch.setattr(podmod, "RunpodRest", _FakeRest)

    async def _noop(self):
        return None

    monkeypatch.setattr(Pod, "_wait_provision", _noop)
    monkeypatch.setattr(Pod, "_wait_ready", _noop)
    # CPU + no TTL keeps _create on the (faked) REST path
    return lambda **kw: PodConfig(compute="cpu", ssh_key=keypair,
                                  stop_after=None, terminate_after=None, **kw)


def test_host_check_rerolls_until_accepted(fake_pod_backend):
    from bellhop.pod import pod

    rejected = []

    async def check(p):
        if p.id == "pod-0":
            rejected.append(p.id)
            raise RuntimeError("blocklisted ip")

    async def go():
        async with pod(fake_pod_backend(host_check=check)) as p:
            return p.id

    assert asyncio.run(go()) == "pod-1"
    assert rejected == ["pod-0"]
    assert _FakeRest.created == ["pod-0", "pod-1"]
    assert "pod-0" in _FakeRest.deleted        # the rejected host was torn down
    assert _FakeRest.deleted.count("pod-1") == 1  # accepted one torn down once, at exit


def test_host_check_exhaustion_raises_with_reasons(fake_pod_backend):
    from bellhop.pod import pod

    async def reject_all(p):
        raise RuntimeError(f"upload probe 6MB/s on {p.id}")

    async def go():
        async with pod(fake_pod_backend(host_check=reject_all, host_check_retries=2)):
            pass

    with pytest.raises(ProvisionError, match="rejected every host") as ei:
        asyncio.run(go())
    assert "pod-0" in str(ei.value) and "pod-2" in str(ei.value)  # every attempt listed
    assert len(_FakeRest.created) == 3                            # 1 + 2 rerolls
    assert set(_FakeRest.created) <= set(_FakeRest.deleted)       # none leaked


def test_host_check_rejected_pod_dies_even_with_keep(fake_pod_backend):
    from bellhop.pod import pod

    async def check(p):
        if p.id == "pod-0":
            raise RuntimeError("bad host")

    async def go():
        async with pod(fake_pod_backend(host_check=check), keep=True) as p:
            return p.id

    assert asyncio.run(go()) == "pod-1"
    assert "pod-0" in _FakeRest.deleted        # rejected: deleted despite keep=True
    assert "pod-1" not in _FakeRest.deleted    # kept: keep=True honored


# --- box.keep: the mid-session escape hatch --------------------------------------

def test_keep_defaults_false_on_every_box():
    from bellhop.modal_box import Sandbox
    from bellhop.nebius_box import NebiusVm

    for cls in (Pod, Sandbox, LambdaInstance, NebiusVm):
        assert cls.keep is False


def test_box_keep_flag_skips_teardown(fake_pod_backend):
    from bellhop.pod import pod

    async def go():
        async with pod(fake_pod_backend()) as p:
            p.keep = True   # "the checkpoint marker exists — keep this box"
            return p.id

    pid = asyncio.run(go())
    assert pid not in _FakeRest.deleted


# --- run(keep_on_failure=...) + failure.log ---------------------------------------

class _FakeBox:
    id = "fake-box"
    keep = False

    def __init__(self, job_exit=1):
        self.job_exit = job_exit

    async def exec(self, cmd, env=None, timeout=None):
        if "tee" in cmd:   # the job script
            return ExecResult(self.job_exit, "job stdout", "rclone: connection reset")
        return ExecResult(0, "", "")

    async def push(self, local, remote):
        pass

    async def exists_remote(self, path):
        return False

    async def teardown(self):
        pass


def _cfg():
    from bellhop import ModalConfig

    return ModalConfig()   # replace()-able stand-in; open_box is faked anyway


def _fake_open_box(box):
    @contextlib.asynccontextmanager
    async def fake(backend, *, keep=False, api_key=None):
        yield box

    return fake


def test_keep_on_failure_flags_box_and_persists_log(tmp_path, monkeypatch):
    import importlib

    runmod = importlib.import_module("bellhop.run")
    box = _FakeBox(job_exit=1)
    monkeypatch.setattr(runmod, "open_box", _fake_open_box(box))
    spec = RunSpec(slug="x", codebase=str(tmp_path), run="python go.py",
                   local_out=str(tmp_path / "out"))

    with pytest.raises(RemoteJobError, match="failure.log"):
        asyncio.run(runmod.run(spec, _cfg(), keep_on_failure=True))
    assert box.keep is True
    log = (tmp_path / "out" / "failure.log").read_text()
    assert "rclone: connection reset" in log     # the FULL fatal string survives
    assert "job stdout" in log


def test_keep_on_failure_off_by_default(tmp_path, monkeypatch):
    import importlib

    runmod = importlib.import_module("bellhop.run")
    box = _FakeBox(job_exit=1)
    monkeypatch.setattr(runmod, "open_box", _fake_open_box(box))
    spec = RunSpec(slug="x", codebase=str(tmp_path), run="python go.py",
                   local_out=str(tmp_path / "out"))

    with pytest.raises(RemoteJobError):
        asyncio.run(runmod.run(spec, _cfg()))
    assert box.keep is False


def test_run_sets_salvage_hook(tmp_path, monkeypatch):
    import importlib

    runmod = importlib.import_module("bellhop.run")
    box = _FakeBox(job_exit=0)

    async def exists(path):
        return True

    async def pull(remote, dest):
        box.pulled = (remote, str(dest))

    box.exists_remote, box.pull = exists, pull
    monkeypatch.setattr(runmod, "open_box", _fake_open_box(box))
    spec = RunSpec(slug="x", codebase=str(tmp_path), run="python go.py",
                   local_out=str(tmp_path / "out"))
    asyncio.run(runmod.run(spec, _cfg()))
    assert callable(box.on_lifetime_expiry)      # the watchdog grace hook is armed
    asyncio.run(box.on_lifetime_expiry())
    assert box.pulled[0].endswith("/results")


# --- watchdog grace: salvage before destruction -----------------------------------

def test_watchdog_runs_grace_hook_before_teardown():
    order = []

    class Box:
        id = "b-1"
        _noun = "vm"
        _lifetime_expired = False

        async def teardown(self):
            order.append("teardown")

    box = Box()

    async def grace():
        order.append("grace")

    box.on_lifetime_expiry = grace
    asyncio.run(lifetime_watchdog(box, timedelta(seconds=0.01)))
    assert order == ["grace", "teardown"]
    assert box._lifetime_expired


def test_watchdog_teardown_proceeds_when_grace_fails(capsys):
    class Box:
        id = "b-1"
        _noun = "vm"
        _lifetime_expired = False
        torn = False

        async def teardown(self):
            self.torn = True

    box = Box()

    async def bad_grace():
        raise RuntimeError("pull wedged")

    box.on_lifetime_expiry = bad_grace
    asyncio.run(lifetime_watchdog(box, timedelta(seconds=0.01)))
    assert box.torn                              # the TTL still wins
    assert "grace hook failed" in capsys.readouterr().err


# --- exec_detached: SIGHUP-proof jobs ----------------------------------------------

class _StubSsh(SshBox):
    """SshBox with scripted transport: records exec cmds, replays _ssh_raw."""

    id = "stub-1"
    ssh_user = "ubuntu"
    _noun = "box"

    def __init__(self, raw_responses):
        self.execs: list[str] = []
        self.raws: list[str] = []
        self._raw_responses = list(raw_responses)

    async def exec(self, cmd, env=None, timeout=None):
        self.execs.append(cmd)
        return ExecResult(0, "", "")

    async def _ssh_raw(self, cmd, timeout=600):
        self.raws.append(cmd)
        return self._raw_responses.pop(0) if self._raw_responses else ExecResult(1, "", "")


def test_exec_detached_start_shape():
    box = _StubSsh([])
    job = asyncio.run(box.exec_detached("python train.py", env={"HF_TOKEN": "t0k"},
                                        name="midtrain"))
    assert isinstance(job, DetachedJob) and job.name == "midtrain"
    start = box.execs[0]
    assert "setsid" in start                       # survives launcher death
    assert "echo $? >" in start                    # exit code is recorded
    assert "export HF_TOKEN=t0k" in start          # env inside the script...
    assert "python train.py" in start
    assert "/tmp/bellhop-jobs/midtrain" in start


def test_exec_detached_rejects_bad_names():
    with pytest.raises(PreflightError, match="job name"):
        asyncio.run(_StubSsh([]).exec_detached("true", name="bad name; rm -rf"))


def test_detached_job_wait_polls_to_exit_code():
    box = _StubSsh([
        ExecResult(1, "", ""),          # exit file absent -> still running
        ExecResult(0, "7\n", ""),       # finished with exit 7
        ExecResult(0, "last log line", ""),  # tail
    ])
    job = DetachedJob(box, "midtrain")
    res = asyncio.run(job.wait(poll=0.01))
    assert res.exit_code == 7
    assert res.stdout == "last log line"


def test_detached_job_wait_timeout_leaves_job_running():
    from bellhop import ExecTimeoutError

    box = _StubSsh([])   # every poll: no exit file yet
    job = DetachedJob(box, "midtrain")
    with pytest.raises(ExecTimeoutError, match="keeps running"):
        asyncio.run(job.wait(poll=0.01, timeout=0.05))


def test_detached_job_running_and_reattach():
    box = _StubSsh([ExecResult(1, "", ""), ExecResult(0, "0\n", "")])
    job = DetachedJob(box, "midtrain")   # reattach-by-name: no exec_detached call
    assert asyncio.run(job.running()) is True
    assert asyncio.run(job.exit_code()) == 0


# --- diagnostics: the 401 that says nothing ----------------------------------------

def test_graphql_401_names_the_key_problem():
    from bellhop.graphql import RunpodGraphQL

    async def handler(req):
        return httpx.Response(401, json={"error": {}})

    gql = RunpodGraphQL(api_key="bad")
    gql._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ProvisionError, match="RUNPOD_API_KEY"):
        asyncio.run(gql._post("query {}", {}))


def test_lambda_rest_pacing_shares_process_budget(monkeypatch):
    # two clients, one budget: the second request waits for the first's window
    import bellhop.lambda_box as lb

    monkeypatch.setattr(LambdaRest, "min_request_interval", 100.0)
    monkeypatch.setattr(lb, "_last_request", 0.0)
    monkeypatch.setattr(lb, "_last_launch", 0.0)
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        monkeypatch.setattr(lb, "_last_request", 0.0)  # unblock after one wait

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def go():
        a, b = LambdaRest(api_key="k"), LambdaRest(api_key="k")
        import time

        monkeypatch.setattr(lb, "_last_request", time.monotonic())  # a just fired
        await b._pace(launch=False)                                 # b must wait

    asyncio.run(go())
    assert sleeps and sleeps[0] > 90