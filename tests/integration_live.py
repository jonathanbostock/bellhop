"""Live end-to-end tests: provision REAL RunPod pods (costs $).

Skipped by default. Run explicitly with:
    RUNPOD_LIVE=1 pytest tests/integration_live.py -s
(needs RUNPOD_API_KEY and an ~/.ssh/id_ed25519 keypair).

Knobs:
    BELLHOP_LIVE_GCS=gs://bucket/prefix   also exercise the GCS upload leg
                                          (needs gcloud; off by default so CI
                                          runners don't need cloud creds)
    BELLHOP_LIVE_GPU=<alias>              override the call() test's GPU

Stock-outs are skips, not failures: RunPod running dry on a GPU type is not
a bellhop regression, and a suite that fails on capacity noise gets ignored.
"""
import asyncio
import os
import time
from datetime import timedelta

import pytest

from bellhop import PodConfig, ProvisionError, RunSpec, SshProbe, is_capacity_error, run

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUNPOD_LIVE"),
    reason="set RUNPOD_LIVE=1 to run the billed live pod test",
)

_TESTCODE = os.path.join(os.path.dirname(__file__), os.pardir, "_testcode")


def _live(coro):
    """asyncio.run, but a capacity-shaped provision failure is a skip."""
    try:
        asyncio.run(coro)
    except ProvisionError as e:
        if is_capacity_error(e):
            pytest.skip(f"RunPod capacity, not a regression: {e}")
        raise


async def _run():
    t0 = time.time()
    spec = RunSpec(
        slug="rpr-selftest",
        codebase=_TESTCODE,
        run="python go.py",
        env={"MY_SECRET": "s3cr3t-xyz"},  # validates env-injection (should appear in out.txt)
        gcs_base=os.environ.get("BELLHOP_LIVE_GCS"),  # upload leg is opt-in
    )
    cfg = PodConfig(
        gpu="RTX4090",   # exercises the canonical-alias path end-to-end
        cloud="COMMUNITY",
        container_disk_gb=20,
        ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=600),
        ready_timeout=timedelta(seconds=600),
    )
    res = await run(spec, cfg)
    print("=== TEST RESULT ===")
    print("elapsed_s:", round(time.time() - t0))
    print("pod_id:", res.pod_id)
    print("remote_exit:", res.remote_exit)
    print("gcs_uri:", res.gcs_uri)
    print("retrieve:", res.retrieve_cmd)
    print("log_tail:\n" + res.log_tail)
    assert res.remote_exit == 0
    assert "MY_SECRET=s3cr3t-xyz" in res.log_tail  # env-injection worked


def test_live_end_to_end():
    _live(_run())


async def _run_call():
    """call() end-to-end on a real pod: parity pre-flight, cloudpickle
    bootstrap, config.pip deps-on-enter, closure round trip, GPU visibility,
    original-type exception re-raise.

    NB the *client* Python minor version must match the image's (pytorch-cuda
    = py3.11) — run this from a 3.11 venv or the parity pre-flight will
    (correctly) refuse.
    """
    from bellhop import RemoteCallError, pod

    t0 = time.time()
    cfg = PodConfig(
        gpu=os.environ.get("BELLHOP_LIVE_GPU", "RTX4090"),  # override on stock-outs
        cloud="COMMUNITY",
        image_preset="pytorch-cuda",         # py3.11 on the box
        pip=["tqdm"],                        # exercises deps-on-enter
        name="bellhop-call-live",
        provision_timeout=timedelta(seconds=600),
        ready_timeout=timedelta(seconds=600),
        max_lifetime=timedelta(hours=1),     # safety backstop for a ~5min test
    )
    factor = 3                               # captured by closure

    def compute(xs, scale=1.0):
        import torch
        import tqdm
        return {"sum": sum(xs) * scale * factor,
                "cuda": torch.cuda.is_available(),
                "tqdm": tqdm.__version__}

    def boom():
        raise ValueError("live kaboom")

    async with pod(cfg) as p:
        out = await p.call(compute, [1, 2, 3], scale=2.0)
        print("call result:", out, "| elapsed_s:", round(time.time() - t0))
        assert out["sum"] == 36.0            # args + closure round-tripped
        assert out["cuda"] is True           # really ran on the GPU box
        assert out["tqdm"]                   # config.pip landed before yield
        try:
            await p.call(boom)
            raise AssertionError("expected ValueError from the box")
        except ValueError as e:
            assert isinstance(e.__cause__, RemoteCallError)
            assert "live kaboom" in e.__cause__.remote_traceback
    print("=== CALL LIVE TEST PASSED === total_s:", round(time.time() - t0))


def test_live_call():
    _live(_run_call())


async def _run_slow_boot():
    """The issue-#27 path end-to-end: non-RunPod image + docker_start_cmd sshd
    bootstrap + default TTL (GraphQL create), carried to readiness by the
    docker_start_cmd-widened default windows — no explicit timeouts here on
    purpose; that IS the assertion."""
    from bellhop.pod import pod

    sshd = (
        'apt-get update && apt-get install -y openssh-server && mkdir -p /run/sshd ~/.ssh'
        ' && echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
        " && /usr/sbin/sshd -D"
    )
    cfg = PodConfig(
        gpu="RTX4090",
        image="ubuntu:22.04",
        docker_start_cmd=sshd,
        container_disk_gb=15,
        name="bellhop-live-slowboot",
    )
    assert cfg.provision_timeout == timedelta(seconds=1200)  # widened default resolved
    assert cfg.has_ttl()  # default timers on -> _gql_create path
    t0 = time.time()
    async with pod(cfg) as p:
        r = await p.exec("echo alive && uname -a")
        print(f"pod {p.id} ready in {time.time() - t0:.0f}s: {r.stdout.strip()}")
        assert r.exit_code == 0
        assert "alive" in r.stdout


def test_live_slow_boot():
    _live(_run_slow_boot())


if __name__ == "__main__":
    asyncio.run(_run())
