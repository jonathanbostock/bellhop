"""M0 live probe for RunPod Instant Clusters (docs/design/instant-clusters.md §6-7).

Creates the cheapest 2-node TRAINING cluster it can get stock for, then answers
the design doc's open questions:

  Q1  are the cluster env vars (PRIMARY_ADDR/NODE_RANK/...) visible over SSH,
      or only in /proc/1/environ?
  Q2  does deleteCluster cascade-delete the member pods?
  Q3  does every member pod get its own public IP + port-22 mapping?
  Q4  what does the account actually get stock/permission for?
  Q5  is a custom imageName accepted on createCluster?  (answered implicitly:
      we pass imageName explicitly rather than a template)

plus the functional core: a torchrun all-reduce across both nodes over ens1.

Everything runs under a finally that deletes the cluster and verifies the pods
are gone (falling back to per-pod REST deletes). Budget: minutes of 2-node
time — dollars, not tens of dollars.

Run:  cd repos/arsenal && .venv/bin/python \
        .claude/worktrees/bellhop-instant-clusters/packages/bellhop/scripts/probe_clusters.py
"""

import asyncio
import json
import pathlib
import re
import shlex
import time
from datetime import timedelta

from bellhop.graphql import GRAPHQL_URL, RunpodGraphQL
from bellhop.pod import Pod, PodConfig
from bellhop.rest import RunpodRest

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# cheapest-first (gpuTypeId, gpuCountPerPod) ladder; podCount is always 2.
# 1-GPU nodes may be rejected (clusters are sold as full nodes in the console)
# — that rejection is itself a Q4 data point, so start there.
LADDER = [
    ("NVIDIA A100 80GB PCIe", 1),
    ("NVIDIA A100-SXM4-80GB", 1),
    ("NVIDIA H100 80GB HBM3", 1),
    ("NVIDIA A100-SXM4-80GB", 2),
    ("NVIDIA H100 80GB HBM3", 2),
    ("NVIDIA A100-SXM4-80GB", 8),
    ("NVIDIA H100 80GB HBM3", 8),
]

CREATE_CLUSTER = """
mutation createCluster($input: CreateClusterInput!) {
  createCluster(input: $input) { id name type gpuTypeId podCount gpuCountPerPod pods { id } }
}
"""
DELETE_CLUSTER = """
mutation deleteCluster($input: DeleteClusterInput!) { deleteCluster(input: $input) }
"""
LIST_CLUSTERS = """
query { myself { clusters { id name pods { id } } } }
"""

ENV_PAT = "PRIMARY|MASTER|NODE_ADDR|NODE_RANK|NUM_NODES|NUM_TRAINERS|WORLD_SIZE|HOST_NODE"

ALLREDUCE_PY = r"""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
t = torch.tensor([float(rank)], device=f"cuda:{int(os.environ['LOCAL_RANK'])}")
dist.all_reduce(t)
expect = world * (world - 1) / 2
print(f"ALLREDUCE rank={rank} world={world} sum={t.item()} expect={expect} "
      f"{'OK' if t.item() == expect else 'MISMATCH'}", flush=True)
dist.destroy_process_group()
"""

# every exec preamble: surface the container's PID-1 env into the ssh session
ENV_PREAMBLE = (
    'while read -r kv; do export "$kv"; done '
    f'< <(tr "\\0" "\\n" < /proc/1/environ | grep -E "^({ENV_PAT})=")'
)


def api_key() -> str:
    cfg = pathlib.Path.home().joinpath(".runpod/config.toml").read_text()
    # NB the value is single-quoted in this file — strip either quote style
    return re.search(r"""apikey\s*=\s*['"]?([^'"\s]+)['"]?""", cfg).group(1)


def report(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}", flush=True)


# cap on what we'll pay per node-hour before skipping a rung (2 nodes ⇒ 2×)
MAX_DEPLOY_COST = 4.0


async def create_cluster(gql: RunpodGraphQL, pubkey: str) -> tuple[dict, int]:
    last_err = None
    for gpu_id, per_pod in LADDER:
        inp = {
            "gpuTypeId": gpu_id,
            "podCount": 2,
            "gpuCountPerPod": per_pod,
            "type": "TRAINING",
            "imageName": IMAGE,
            "containerDiskInGb": 20,
            "ports": "22/tcp",
            "startSsh": True,
            "env": [{"key": "PUBLIC_KEY", "value": pubkey}],
        }
        # two attempts per rung: without deployCost the error leaks the current
        # minimum per-node $/hr ("requested price 0 is less than the current
        # minimum price (1.39)"); second attempt pays exactly that.
        for attempt in range(2):
            report("create", f"trying {gpu_id} x{per_pod}/node "
                             f"(deployCost={inp.get('deployCost', 'unset')}) ...")
            try:
                data = await gql._post(CREATE_CLUSTER, {"input": inp})
                clu = data["createCluster"]
                report("create", f"OK id={clu['id']} name={clu['name']} "
                                 f"pods={[p['id'] for p in clu['pods']]} "
                                 f"deployCost={inp.get('deployCost')}/node-hr")
                return clu, per_pod
            except Exception as e:  # noqa: BLE001 — ladder walks past stock/rule rejections
                last_err = e
                report("create", f"rejected: {e}")
                m = re.search(r"minimum price \(([\d.]+)\)", str(e))
                # the server divides deployCost by podCount before comparing to
                # the per-node minimum (bid 1.39 → "requested price 0.695"),
                # so deployCost prices the whole cluster
                if attempt == 0 and m and float(m.group(1)) <= MAX_DEPLOY_COST:
                    inp["deployCost"] = round(float(m.group(1)) * inp["podCount"], 2)
                    continue
                if m and float(m.group(1)) > MAX_DEPLOY_COST:
                    report("create", f"skipping rung: min {m.group(1)}/node-hr > cap {MAX_DEPLOY_COST}")
                break
    raise SystemExit(f"Q4: no ladder rung accepted — last error: {last_err}")


async def main() -> None:
    key = api_key()
    pubkey = pathlib.Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()
    findings: dict[str, str] = {}
    t0 = time.monotonic()

    async with RunpodGraphQL(key) as gql, RunpodRest(key) as rest:
        clu, per_pod = await create_cluster(gql, pubkey)
        cluster_id = clu["id"]
        pod_ids = [p["id"] for p in clu["pods"]]
        cfg = PodConfig(gpu="A100", gpu_count=per_pod,
                        provision_timeout=timedelta(seconds=900),
                        ready_timeout=timedelta(seconds=900))
        pods = [Pod(rest, pid, cfg) for pid in pod_ids]
        try:
            # ---- Q3: per-pod public IP + ssh ------------------------------
            report("provision", f"waiting for {len(pods)} pods to get IP+port22 ...")
            await asyncio.gather(*(p._wait_provision() for p in pods))
            findings["Q3"] = "YES — " + "; ".join(
                f"{p.id}: {p.host}:{p.mapped_port(22)}" for p in pods)
            report("provision", findings["Q3"])
            await asyncio.gather(*(p._wait_ready() for p in pods))
            report("ready", f"ssh up on both pods ({time.monotonic()-t0:.0f}s from create)")

            # ---- Q1: env visibility --------------------------------------
            # first probe found PRIMARY_ADDR/PRIMARY_PORT/MASTER_* absent even
            # in PID-1 env (only NODE_RANK/NODE_ADDR/NUM_*/WORLD_SIZE present),
            # so also check /etc/rp_environment and /etc/environment
            probe = (
                f'echo "SSH_SESSION:"; env | grep -E "^({ENV_PAT})=" || echo "  (none)"; '
                f'echo "PID1:"; tr "\\0" "\\n" < /proc/1/environ | grep -E "^({ENV_PAT})=" || echo "  (none)"; '
                f'echo "RP_ENV_FILE:"; grep -hE "({ENV_PAT})" /etc/rp_environment /etc/environment 2>/dev/null || echo "  (none)"'
            )
            envs = await asyncio.gather(*(p.exec(probe, timeout=60) for p in pods))
            ranks: dict[str, int] = {}
            node_ips: dict[int, str] = {}
            for p, r in zip(pods, envs):
                report("env", f"pod {p.id}:\n{r.stdout}")
                m = re.search(r"NODE_RANK=(\d+)", r.stdout)
                if m:
                    ranks[p.id] = int(m.group(1))
                a = re.search(r"NODE_ADDR=([\d.]+)", r.stdout)  # strips the /24 suffix
                if m and a:
                    node_ips[int(m.group(1))] = a.group(1)
                ssh_part = r.stdout.split("PID1:")[0]
                findings.setdefault(
                    "Q1",
                    "ssh session inherits cluster env"
                    if "NODE_RANK=" in ssh_part else
                    "NOT in ssh session; present in /proc/1/environ (use preamble); "
                    "PRIMARY_ADDR/PORT absent everywhere — derive from rank-0 NODE_ADDR")
            if len(ranks) < 2:
                findings["Q1"] += " — WARNING: NODE_RANK not found on both pods"
            report("ranks", f"{ranks} ips={node_ips}")

            # ---- functional: all-reduce over the overlay network ---------
            # rendezvous is self-derived: rank-0's NODE_ADDR + a fixed port —
            # more robust than trusting PRIMARY_ADDR injection anyway
            primary_ip = node_ips[0]
            script = (
                f"{ENV_PREAMBLE}\n"
                "export NCCL_SOCKET_IFNAME=ens1 NCCL_DEBUG=WARN\n"
                f"cat > /tmp/allreduce.py <<'PYEOF'\n{ALLREDUCE_PY}\nPYEOF\n"
                f"torchrun --nnodes 2 --node_rank $NODE_RANK --nproc_per_node {per_pod} "
                "--rdzv_id m0probe --rdzv_backend static "
                f'--rdzv_endpoint "{primary_ip}:29500" /tmp/allreduce.py'
            )
            report("allreduce", f"launching torchrun on both nodes (rdzv {primary_ip}:29500) ...")
            results = await asyncio.gather(
                *(p.exec(script, timeout=600) for p in pods), return_exceptions=True)
            ok = True
            for p, r in zip(pods, results):
                if isinstance(r, Exception):
                    ok = False
                    report("allreduce", f"pod {p.id} EXC: {r}")
                else:
                    ok = ok and r.exit_code == 0 and "MISMATCH" not in r.stdout
                    report("allreduce", f"pod {p.id} exit={r.exit_code}\n{r.stdout}\n{r.stderr[-2000:]}")
            findings["allreduce"] = "OK" if ok else "FAILED (see log)"
        finally:
            # ---- Q2 + teardown -------------------------------------------
            report("teardown", f"deleteCluster {cluster_id}")
            try:
                await gql._post(DELETE_CLUSTER, {"input": {"id": cluster_id}})
            except Exception as e:  # noqa: BLE001
                report("teardown", f"deleteCluster failed: {e}")
            await asyncio.sleep(10)
            survivors = []
            for pid in pod_ids:
                try:
                    meta = await rest.get_pod(pid)
                    survivors.append((pid, meta.get("desiredStatus")))
                except Exception:  # 404 = gone, which is what we want
                    pass
            findings["Q2"] = ("cascade YES — all pods gone" if not survivors
                              else f"cascade NO — survivors {survivors}, deleting individually")
            for pid, _ in survivors:
                try:
                    await rest.delete_pod(pid)
                    report("teardown", f"deleted straggler pod {pid}")
                except Exception as e:  # noqa: BLE001
                    report("teardown", f"FAILED to delete pod {pid}: {e} — DELETE IT IN THE CONSOLE")
            data = await gql._post(LIST_CLUSTERS, {})
            leftover = [c["id"] for c in data["myself"]["clusters"]]
            report("teardown", f"clusters remaining on account: {leftover or 'none'}")

    findings["Q4"] = f"accepted rung: {clu['gpuTypeId']} x{clu['gpuCountPerPod']}/node x{clu['podCount']} nodes"
    findings["Q5"] = f"custom imageName accepted: {IMAGE}"
    findings["wall"] = f"{time.monotonic()-t0:.0f}s total"
    print("\n===== M0 FINDINGS =====")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
