# Bellhop × RunPod Instant Clusters — multi-node design

**Status:** design mockup (nothing implemented). **Driver:** full-param dense
~100B training for science-of-midtraining needs 2–4 coordinated nodes; bellhop
is strictly one-pod-per-job today.

---

## 1. What RunPod Instant Clusters actually provide (verified 2026-08-10)

Everything in this section was verified against the live API with the account
key (read-only probes; nothing was provisioned) or the official docs — not
from marketing pages.

### 1.1 The API is GraphQL-only

The public REST API (`rest.runpod.io/v1`, the one bellhop uses for pods) has
**no cluster endpoints** — confirmed by dumping its OpenAPI spec. The v2 BETA
REST API has cluster *billing* history only. `runpodctl` has no cluster
command. Cluster CRUD lives on `https://api.runpod.io/graphql` (same Bearer
auth bellhop's `RunpodGraphQL` already uses for TTL pods).

Schema recovered via validation-error probing (introspection is disabled):

```graphql
mutation createCluster($input: CreateClusterInput!) {
  createCluster(input: $input) { id name pods { id } }
}

# CreateClusterInput — required:
#   gpuTypeId:      String!        # e.g. "NVIDIA H200"
#   podCount:       Int!           # nodes, 2–8
#   gpuCountPerPod: Int!           # 8 for a full node
#   type:           ClusterType!   # TRAINING | SLURM | RAY
# optional:
#   deployCost:     Float          # REQUIRED in practice: $/hr for the WHOLE
#                                  # cluster; server divides by podCount and
#                                  # compares to the per-node minimum. Omitting
#                                  # it bids 0 → rejected with an error that
#                                  # leaks the current minimum ("requested
#                                  # price 0 is less than the current minimum
#                                  # price (3.29)") — bid min × podCount.
#   templateId, imageName: String
#   env: [EnvironmentVariableInput]     # {key, value}
#   dataCenterId: String
#   networkVolumeId: String
#   containerDiskInGb, volumeInGb: Int
#   volumeMountPath, dockerArgs: String
#   ports: String
#   startSsh: Boolean
#   allowedCudaVersions: [String]

mutation { deleteCluster(input: { id: "<clusterId>" }) }

query { myself { clusters {
  id name type gpuTypeId podCount gpuCountPerPod createdAt userId
  pods { id }        # ordinary Pod objects → existing REST GET /pods/{id}
                     # works for port mappings / public IP per node
} } }
```

Notes:

- `ClusterType.TRAINING` is the generic/PyTorch flavor (the console's
  default); `SLURM` and `RAY` are managed variants we don't need.
- **No TTL fields.** `stopAfter` / `terminateAfter` do not exist on
  `CreateClusterInput`. Bellhop's native-TTL safety net (`max_lifetime`) has
  no server-side equivalent for clusters — teardown reliability must be owned
  client-side (§5).
- No `name` input either (server-assigned); pods appear as
  `CLUSTERNAME-pod-<rank>`.

### 1.2 Node environment contract

Injected automatically on every node
([configuration reference](https://docs.runpod.io/instant-clusters/configuration)):

| Var | Meaning |
| --- | --- |
| `PRIMARY_ADDR` / `MASTER_ADDR` | primary node's static overlay IP |
| `PRIMARY_PORT` / `MASTER_PORT` | rendezvous port |
| `NODE_ADDR` | this node's static overlay IP |
| `NODE_RANK` | 0 = primary |
| `NUM_NODES`, `NUM_TRAINERS` | nodes, GPUs-per-node |
| `WORLD_SIZE` | `NUM_NODES × NUM_TRAINERS` |
| `HOST_NODE_ADDR` | `PRIMARY_ADDR:PRIMARY_PORT` |

Hard requirements from the docs:

- `NCCL_SOCKET_IFNAME=ens1` — inter-node traffic must use the high-bandwidth
  overlay NICs (`ens1`–`ens8`), never `eth0` (172.x = internet only; using it
  ⇒ connection timeouts).
- `torchrun` must use `--rdzv_backend static` — the dynamic `c10d` backend is
  **not supported**.
- RunPod's own axolotl-on-cluster recipe is exactly:

  ```sh
  torchrun --nnodes $NUM_NODES --node_rank $NODE_RANK \
           --nproc_per_node $NUM_TRAINERS \
           --rdzv_id job --rdzv_backend static \
           --rdzv_endpoint "$PRIMARY_ADDR:$PRIMARY_PORT" \
           -m axolotl.cli.train config.yml
  ```

### 1.3 Supported hardware

| GPU | interconnect | scale |
| --- | --- | --- |
| B200 | 3200 Gbps | 2–8 nodes (16–64 GPUs) |
| H200 | 3200 Gbps | 2–8 nodes |
| H100 | 3200 Gbps | 2–8 nodes |
| A100 | 1600 Gbps | 2–8 nodes |

&gt;8 nodes (→512 GPUs) via sales. Accounts have a default cluster spending
limit (raise via help@runpod.io) — check ours before the first real run.
Grafana observability is built in per cluster.

---

## 2. Does 100B dense actually fit? (sizing)

Full-param AdamW mixed precision ≈ 16 B/param (bf16 params + grads, fp32
master + m + v), fully sharded (FSDP2 `FULL_SHARD` across all ranks):

| Model | state | 2×8 H200 (2.26 TB) | 4×8 H200 (4.51 TB) | 2×8 B200 (3.07 TB) |
| --- | --- | --- | --- | --- |
| 70B (Llama-3.1) | 1.12 TB | 70 GB/GPU — OK | 35 GB/GPU — roomy | 70 GB/GPU — roomy |
| ~110–123B (Command-A / Mistral-Large) | 1.8–2.0 TB | 112–123 GB/GPU — **too tight** | 56–62 GB/GPU — OK | 112–123 GB/GPU — OK w/ care |
| 180B (Falcon) | 2.9 TB | no | 90 GB/GPU — tight | no |

So: **2 nodes H200 covers 70B-class; true 100B+ dense wants 4×H200 or
2×B200.** (Caveat worth deciding early: open *dense* models at 100B+ are
thin on the ground — Command-A 111B, Mistral-Large 123B (research license),
Falcon-180B. Most modern 100B+ opens are MoE, which is the ms-swift/Megatron
path, not this one.)

Throughput anchor: 16×H200 at ~40% MFU ≈ 6.3 PFLOP/s ⇒ a 100B model sees
~1B tokens/day (6·N·D); 32 GPUs halves that. Cost anchor: H200 ≈ $4/GPU·hr ⇒
~$64/hr (16), ~$128/hr (32) — verify against the cluster pricing page, which
may differ from pod pricing (billing includes an explicit inter-node
networking component).

---

## 3. Bellhop design

### 3.1 Shape

One new module `src/bellhop/cluster.py`, same idioms as `pod.py`. No new
arsenal package. Public surface:

```python
from bellhop import ClusterConfig, cluster, run_cluster
```

### 3.2 `ClusterConfig`

```python
@dataclass
class ClusterConfig:
    gpu: str                          # canonical alias (H200, B200, …) via GPU_ALIASES
    nodes: int = 2                    # → podCount   (RunPod: 2–8)
    gpu_count: int = 8                # → gpuCountPerPod
    image: str | None = None          # or image_preset, as PodConfig
    env: dict[str, str] = field(default_factory=dict)
    container_disk_gb: int = 100
    volume_gb: int = 0
    network_volume_id: str | None = None   # shared HF cache / dataset / ckpts
    data_center_id: str | None = None      # must match the network volume's DC
    ports: list[str] = field(default_factory=lambda: ["22/tcp"])
    docker_start_cmd: str | None = None
    allowed_cuda_versions: list[str] | None = None
    ready: ReadyProbe = SshProbe("true")   # per-node; barrier applied by cluster()
    provision_timeout: float = 900.0
    ready_timeout: float = 900.0
    poll_interval: float = 8.0
    max_lifetime: float = 24 * 3600.0      # CLIENT-SIDE watchdog — no native TTL!
```

`cluster_type` is not exposed: always `TRAINING`.

### 3.3 `Cluster` box

A `Cluster` is N `Pod`s plus cluster identity — it deliberately reuses the
existing `Pod` channel (SSH exec / tar push / pull) per node, so everything
already proven in `pod.py` carries over:

```python
class Cluster:
    id: str
    nodes: list[Pod]                  # index == NODE_RANK

    @property
    def primary(self) -> Pod: return self.nodes[0]

    async def exec_all(self, script, *, env=None, timeout=None) -> list[ExecResult]:
        # asyncio.gather over nodes; raises ClusterExecError carrying
        # per-node results if any rank fails
    async def push_all(self, local, remote) -> None      # gather push to every node
    async def pull(self, remote, local, *, rank=0)       # results come from rank 0
    async def teardown(self) -> None                     # deleteCluster + verify pods gone
```

### 3.4 `cluster()` context manager (mirrors `pod()`)

```python
@asynccontextmanager
async def cluster(config: ClusterConfig, *, api_key=None):
    gql = RunpodGraphQL(api_key)
    created = await gql.create_cluster(config.to_graphql_input())   # one mutation
    try:
        pods = await _wait_cluster_provision(created["id"])
        # per-pod: same REST polling as _wait_provision (public IP + port 22)
        ranks = await _discover_ranks(pods)
        # SSH sessions do NOT necessarily inherit container env — read
        # NODE_RANK from /proc/1/environ per pod and order nodes[] by it (§6 Q1)
        clu = Cluster(id=created["id"], nodes=ranks)
        await _barrier_ready(clu, config)      # ALL per-node probes must pass
        await _install_pip_all(clu, config)
        watchdog = asyncio.create_task(_lifetime_watchdog(clu, config.max_lifetime))
        try:
            yield clu
        finally:
            watchdog.cancel()
    finally:
        await _teardown_cluster(gql, created["id"])   # always; belt-and-braces §5
```

### 3.5 `run_cluster()` — the one-shot pipeline (mirrors `run()`)

```python
async def run_cluster(spec: RunSpec, config: ClusterConfig, *, api_key=None) -> RunResult:
    async with cluster(config, api_key=api_key) as clu:
        await clu.push_all(spec.codebase, remote_root)
        await clu.exec_all(spec.setup, env=spec.env)          # setup on every node
        results = await clu.exec_all(spec.run, env=spec.env)  # SAME command, every node;
            # the command itself reads NODE_RANK/PRIMARY_ADDR/… from the injected env
        await clu.pull(f"{remote_root}/{spec.results_subdir}", spec.local_out)  # rank 0
    return RunResult(...)
```

Failure semantics for the run stage: `exec_all` runs all ranks under
`asyncio.gather(return_exceptions=True)`; first non-zero exit **cancels the
sibling execs** (with static rendezvous, surviving ranks would otherwise hang
at the next collective until timeout) and raises with rank-0's tail plus the
failing rank's tail. Rank-0 stdout is streamed to the caller (this is where
scimt's loss-divergence guard attaches); other ranks stream to per-rank log
files pulled on failure.

---

## 4. scimt integration sketch (science-of-midtraining)

Minimal-diff plan against `src/scimt/train/axolotl.py`:

1. `PodSpec` gains `nodes: int = 1`. `executor_for(stage)`: `nodes > 1` →
   `ClusterExecutor` (else unchanged).
2. `ClusterExecutor.run_stage` = today's `BellhopExecutor.run_stage` with
   `bellhop.run(...)` → `bellhop.run_cluster(...)` and the run command wrapped:

   ```sh
   export NCCL_SOCKET_IFNAME=ens1
   export NCCL_NVLS_ENABLE=0            # keep existing community-host guard
   torchrun --nnodes "$NUM_NODES" --node_rank "$NODE_RANK" \
            --nproc_per_node "$NUM_TRAINERS" \
            --rdzv_id scimt --rdzv_backend static \
            --rdzv_endpoint "$PRIMARY_ADDR:$PRIMARY_PORT" \
            -m axolotl.cli.train <rendered.yaml>
   ```

   where `PRIMARY_ADDR`/`PRIMARY_PORT` are **injected by bellhop's
   `exec_all`** (rank-0 `NODE_ADDR` + fixed port), not read from RunPod —
   M0 showed RunPod doesn't actually inject them (§6 Q1).

   (today the stage runs bare `axolotl train`, which launches
   single-node accelerate internally).
3. Stage-template changes for 100B:
   - `fsdp_config.state_dict_type: SHARDED_STATE_DICT` — `FULL_STATE_DICT`
     at 100B gathers ~200 GB onto rank-0 CPU per save. The
     `CheckpointSchedulePlugin` keeps the schedule but saves DCP shards;
     consolidation to an HF-loadable checkpoint becomes an offline step
     (accelerate's merge utility) on rank 0 before the GCS push.
   - Checkpoint bus stays `gcs` (rclone), rank 0 only, unchanged pointer
     rows in `checkpoints.jsonl`.
   - `container_disk_gb` sized for ≥2 sharded checkpoints (~400 GB each at
     100B) or checkpoints written to the attached `networkVolumeId`.
4. Data/model staging: attach a **network volume** (same DC as the cluster)
   holding the HF model snapshot + pre-tokenized dataset
   (`axolotl preprocess` once) so N nodes don't each download 200 GB from HF
   and tokenization is bit-identical across ranks.

---

## 5. Teardown safety (no native TTL)

Clusters bill per node-hour with no server-side kill switch, so a leaked
8×H200×4-node cluster is ~$128/hr until noticed. Layers:

1. `cluster()`'s `finally` always calls `deleteCluster`, then polls REST
   `/pods` to confirm every member pod is gone (verify cascade in M0 — if
   `deleteCluster` doesn't cascade, delete pods individually via existing
   REST `delete_pod`).
2. Client-side `max_lifetime` watchdog task inside `cluster()`.
3. A devbox-side reaper: `bellhop clusters gc` CLI subcommand — list
   `myself.clusters`, kill any older than a threshold. Cheap to add and
   covers the "devbox process died" hole that native TTL used to cover.
   (Cron-friendly.)

---

## 6. Open questions — ANSWERED by M0 probe (2026-08-10, live)

`scripts/probe_clusters.py` ran the full cycle three times (~$0.40 total);
final run: create → both pods SSH-routable in 82 s → env probe → cross-node
NCCL all-reduce **OK** → `deleteCluster` → account clean. 103 s wall.

1. **Env-var visibility over SSH: NOT inherited** — cluster env lives only in
   `/proc/1/environ` (no `/etc/rp_environment` on cluster pods). Exec
   preamble required:
   `while read -r kv; do export "$kv"; done < <(tr '\0' '\n' < /proc/1/environ | grep -E '^(NODE_|NUM_|WORLD_)')`.
   **Bigger finding: `PRIMARY_ADDR`/`PRIMARY_PORT`/`MASTER_*` are not
   injected at all** (docs notwithstanding) — only `NODE_RANK`, `NODE_ADDR`
   (CIDR-suffixed, e.g. `10.65.0.2/24`), `NUM_NODES`, `NUM_TRAINERS`,
   `WORLD_SIZE`. Bellhop must derive the rendezvous itself: rank-0's
   `NODE_ADDR` stripped of `/24` + a fixed port (29500). Verified working;
   also more robust than trusting injection.
2. **`deleteCluster` cascade: YES** — member pods gone within ~10 s, nothing
   to clean up individually.
3. **Per-node public IP + SSH: YES** — each member pod gets its own
   `publicIp` + port-22 mapping; bellhop's existing `Pod` channel works
   unmodified per node.
4. **Availability/pricing (this account, EU-ish stock at probe time):**
   cluster minimums leaked by the pricing error: A100-PCIe $1.39, A100-SXM
   $1.59, H100-SXM $3.29 per GPU·hr (≈ pod on-demand rates). A100 2-node
   stock was absent ("Insufficient resources"); H100 2-node available
   instantly. 1-GPU-per-node clusters ARE allowed (cheap smoke tests!).
   H200/B200 rungs not yet probed.
5. **Custom `imageName`: accepted** (probe used `runpod/pytorch:2.4.0`; the
   `PUBLIC_KEY` env → sshd path works exactly as on single pods). Note the
   cluster `name` is server-derived (from the image; there is no name input).

## 7. Milestones

- **M0 — probe. DONE 2026-08-10** (`scripts/probe_clusters.py`, findings in
  §6). Rerun any time; a full cycle costs ~$0.20 and 2 minutes.
- **M1 — bellhop `cluster.py`. DONE 2026-08-10** (bellhop 0.8.0):
  `ClusterConfig` / `Cluster` / `cluster()` / `run_cluster` /
  `list_clusters` / `gc_clusters` + `bellhop clusters list|gc` CLI, 13
  offline tests, and a live e2e (`scripts/e2e_cluster.py`) that **passed in
  58 s** — full pipeline (auto-bid create → rank discovery → push_all →
  torchrun all-reduce off injected env → rank-0 pull → cascade teardown)
  on a real 2-node H100 cluster.
- **M2 — scimt `ClusterExecutor` + smoke.** Gemma-3-12B midtrain stage on a
  2-node H100/H200 cluster; acceptance = loss curve matches the single-node
  baseline run at equal effective global batch, and checkpoints land on GCS.
- **M3 — 100B run.** Pick the dense target model (§2 caveat), 4×H200 (or
  2×B200), `SHARDED_STATE_DICT` stage template, network-volume staging,
  budget from §2 anchors.
