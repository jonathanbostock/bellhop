"""Live e2e for bellhop.run_cluster — the M1 acceptance gate.

Drives the full public API against a real 2-node 1-GPU-per-node cluster
(~$7/hr held for ~2 min): push a tiny codebase to both nodes, torchrun an
all-reduce using ONLY the env bellhop injects, pull results from rank 0,
verify the cascade teardown. Compare scripts/probe_clusters.py, which did the
same with hand-rolled GraphQL — this exercises ClusterConfig/cluster()/
run_cluster end to end.

Run:  cd repos/arsenal && .venv/bin/python \
        .claude/worktrees/bellhop-instant-clusters/packages/bellhop/scripts/e2e_cluster.py
"""

import asyncio
import json
import pathlib
import re
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bellhop import ClusterConfig, RunSpec, list_clusters, run_cluster  # noqa: E402

TRAIN_PY = r"""
import json, os, pathlib, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
t = torch.tensor([float(rank)], device=f"cuda:{int(os.environ['LOCAL_RANK'])}")
dist.all_reduce(t)
expect = world * (world - 1) / 2
print(f"rank={rank} sum={t.item()} expect={expect}", flush=True)
assert t.item() == expect, "all-reduce mismatch"
if rank == 0:
    pathlib.Path("results").mkdir(exist_ok=True)
    json.dump({"world_size": world, "allreduce_sum": t.item(),
               "node_rank": os.environ["NODE_RANK"],
               "primary_addr": os.environ["PRIMARY_ADDR"]},
              open("results/allreduce.json", "w"))
dist.destroy_process_group()
"""

RUN_CMD = (
    'torchrun --nnodes "$NUM_NODES" --node_rank "$NODE_RANK" '
    '--nproc_per_node "$NUM_TRAINERS" --rdzv_id e2e --rdzv_backend static '
    '--rdzv_endpoint "$PRIMARY_ADDR:$PRIMARY_PORT" train.py'
)


def api_key() -> str:
    cfg = pathlib.Path.home().joinpath(".runpod/config.toml").read_text()
    return re.search(r"""apikey\s*=\s*['"]?([^'"\s]+)['"]?""", cfg).group(1)


async def main() -> None:
    key = api_key()
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        (pathlib.Path(td) / "train.py").write_text(TRAIN_PY)
        out = tempfile.mkdtemp(prefix="bellhop-e2e-")
        spec = RunSpec(slug="cluster-e2e", codebase=td, run=RUN_CMD,
                       results_subdir="results", local_out=out, gcs_base=None)
        config = ClusterConfig(
            gpu="H100", nodes=2, gpu_count=1,
            image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            container_disk_gb=20, max_hourly_cost=8.0,
        )
        res = await run_cluster(spec, config, api_key=key)
        print(f"\nrun_cluster returned: cluster={res.pod_id} exit={res.remote_exit} "
              f"({time.monotonic()-t0:.0f}s)")
        print("log tail:\n" + res.log_tail)
        payload = json.load(open(pathlib.Path(res.local_results) / "results" / "allreduce.json"))
        print("pulled results/allreduce.json:", payload)
        assert payload["allreduce_sum"] == 1.0 and payload["world_size"] == 2

    leftover = [c["id"] for c in await list_clusters(key)]
    print("clusters remaining on account:", leftover or "none")
    assert not leftover, "teardown left a cluster behind!"
    print(f"\nE2E PASSED in {time.monotonic()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
