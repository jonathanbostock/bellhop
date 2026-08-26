"""Live end-to-end tests: launch REAL Lambda Cloud instances (costs $).

Skipped by default. Run explicitly with:
    LAMBDA_LIVE=1 pytest tests/integration_lambda.py -s
(needs LAMBDA_API_KEY and an ~/.ssh/id_ed25519 keypair).

Knobs:
    BELLHOP_LIVE_LAMBDA_GPU=<alias>     override the GPU (default A10 —
                                        Lambda's cheapest type)
    BELLHOP_LIVE_LAMBDA_REGION=<name>   pin a region

Stock-outs are skips, not failures: Lambda running dry on a GPU type is not
a bellhop regression, and a suite that fails on capacity noise gets ignored.
NB Lambda bills by the minute with NO server-side TTL — every path here
tears down via the context manager, and a crashed run is visible to
`bellhop lambda gc`.
"""
import asyncio
import os
import time
from datetime import timedelta

import pytest

from bellhop import LambdaConfig, ProvisionError, RunSpec, is_capacity_error, run

pytestmark = pytest.mark.skipif(
    not os.environ.get("LAMBDA_LIVE"),
    reason="set LAMBDA_LIVE=1 to run the billed live Lambda test",
)

_TESTCODE = os.path.join(os.path.dirname(__file__), os.pardir, "_testcode")


def _cfg(**kw) -> LambdaConfig:
    return LambdaConfig(
        gpu=os.environ.get("BELLHOP_LIVE_LAMBDA_GPU", "A10"),
        region=os.environ.get("BELLHOP_LIVE_LAMBDA_REGION"),
        max_lifetime=timedelta(hours=1),   # watchdog backstop for a ~10min test
        **kw,
    )


def _live(coro):
    """asyncio.run, but a capacity-shaped provision failure is a skip."""
    try:
        asyncio.run(coro)
    except ProvisionError as e:
        if is_capacity_error(e):
            pytest.skip(f"Lambda capacity, not a regression: {e}")
        raise


async def _run():
    t0 = time.time()
    spec = RunSpec(
        slug="lambda-selftest",
        codebase=_TESTCODE,
        run="python3 go.py",
        env={"MY_SECRET": "s3cr3t-xyz"},   # validates env-injection
        gcs_base=None,
    )
    res = await run(spec, _cfg())
    print("=== TEST RESULT ===")
    print("elapsed_s:", round(time.time() - t0))
    print("instance_id:", res.pod_id)
    print("remote_exit:", res.remote_exit)
    print("log_tail:\n" + res.log_tail)
    assert res.remote_exit == 0
    assert "MY_SECRET=s3cr3t-xyz" in res.log_tail   # env-injection worked


def test_live_end_to_end():
    _live(_run())


async def _run_box():
    """The composable path: workspace bootstrap, GPU visibility, direct-port
    probes, exec/push/pull parity with the RunPod backend."""
    from bellhop import instance

    t0 = time.time()
    async with instance(_cfg(name="bellhop-lambda-live")) as inst:
        r = await inst.exec("test -w /workspace && echo writable && nvidia-smi -L")
        print(f"instance {inst.id} ready in {time.time() - t0:.0f}s: {r.stdout.strip()}")
        assert r.exit_code == 0
        assert "writable" in r.stdout   # the sudo /workspace bootstrap landed
        assert "GPU" in r.stdout        # really a GPU box
        assert await inst.exists_remote("/workspace")
    print("=== BOX LIVE TEST PASSED === total_s:", round(time.time() - t0))


def test_live_box():
    _live(_run_box())
