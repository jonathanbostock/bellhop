"""Live end-to-end tests: create REAL Nebius VMs (costs $).

Skipped by default. Run explicitly with:
    NEBIUS_LIVE=1 pytest tests/integration_nebius.py -s
(needs Nebius auth — NEBIUS_IAM_TOKEN or a credentials file / CLI config —
plus NEBIUS_PROJECT_ID and an ~/.ssh/id_ed25519 keypair).

Knobs:
    BELLHOP_LIVE_NEBIUS_GPU=<alias>     GPU to use; UNSET by default — the
                                        default run is a cheap cpu-d3 VM,
                                        which exercises every code path
                                        except the GPU platform itself
    BELLHOP_LIVE_NEBIUS_PRESET=<id>     explicit preset override

Stock-outs / quota exhaustion are skips, not failures. NB Nebius has NO
server-side TTL — every path here tears down via the context manager, and a
crashed run is visible to `bellhop nebius gc`.
"""
import asyncio
import os
import time
from datetime import timedelta

import pytest

from bellhop import NebiusConfig, ProvisionError, RunSpec, is_capacity_error, run

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEBIUS_LIVE"),
    reason="set NEBIUS_LIVE=1 to run the billed live Nebius test",
)

_TESTCODE = os.path.join(os.path.dirname(__file__), os.pardir, "_testcode")


def _cfg(**kw) -> NebiusConfig:
    return NebiusConfig(
        gpu=os.environ.get("BELLHOP_LIVE_NEBIUS_GPU"),          # default: CPU box
        preset=os.environ.get("BELLHOP_LIVE_NEBIUS_PRESET"),
        max_lifetime=timedelta(hours=1),   # watchdog backstop for a ~10min test
        **kw,
    )


def _live(coro):
    """asyncio.run, but a capacity-shaped provision failure is a skip."""
    try:
        asyncio.run(coro)
    except ProvisionError as e:
        if is_capacity_error(e):
            pytest.skip(f"Nebius capacity/quota, not a regression: {e}")
        raise


async def _run():
    t0 = time.time()
    spec = RunSpec(
        slug="nebius-selftest",
        codebase=_TESTCODE,
        run="python3 go.py",
        env={"MY_SECRET": "s3cr3t-xyz"},   # validates env-injection
        gcs_base=None,
    )
    res = await run(spec, _cfg())
    print("=== TEST RESULT ===")
    print("elapsed_s:", round(time.time() - t0))
    print("vm_id:", res.pod_id)
    print("remote_exit:", res.remote_exit)
    print("log_tail:\n" + res.log_tail)
    assert res.remote_exit == 0
    assert "MY_SECRET=s3cr3t-xyz" in res.log_tail   # env-injection worked


def test_live_end_to_end():
    _live(_run())


async def _run_box():
    """The composable path: cloud-init user, workspace bootstrap, direct-port
    exec/push/pull parity with the other SSH backends."""
    from bellhop import vm

    t0 = time.time()
    async with vm(_cfg(name="bellhop-nebius-live")) as box:
        r = await box.exec("test -w /workspace && echo writable && id -un && uname -a")
        print(f"vm {box.id} ready in {time.time() - t0:.0f}s: {r.stdout.strip()}")
        assert r.exit_code == 0
        assert "writable" in r.stdout   # the sudo /workspace bootstrap landed
        assert "ubuntu" in r.stdout     # the cloud-init user we asked for
        assert await box.exists_remote("/workspace")
        if os.environ.get("BELLHOP_LIVE_NEBIUS_GPU"):
            g = await box.exec("nvidia-smi -L")
            assert g.exit_code == 0 and "GPU" in g.stdout
    print("=== BOX LIVE TEST PASSED === total_s:", round(time.time() - t0))


def test_live_box():
    _live(_run_box())
