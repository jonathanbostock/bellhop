# bellhop

[![CI](https://github.com/jonathanbostock/bellhop/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jonathanbostock/bellhop/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Run your code on a disposable cloud GPU — provision, execute, bring results
back, tear down.** Async Python, four interchangeable backends:
[RunPod](https://runpod.io) pods, [Modal](https://modal.com) sandboxes,
[Lambda Cloud](https://lambda.ai) instances and [Nebius](https://nebius.com)
VMs. Scales from one CPU box to a multi-node H200 cluster without changing
shape.

> This is a fork of [dtch1997's bellhop](https://github.com/dtch1997/arsenal/tree/main/packages/bellhop)
> that adds the **Lambda Cloud** and **Nebius** backends — providers with
> fixed on-demand pricing and real capacity signals, for when RunPod's spot
> market is the wrong trade.

```python
from bellhop import pod, PodConfig

async with pod(PodConfig(gpu="H100")) as p:
    await p.push("./mycode", "/workspace/job")
    await p.exec("cd /workspace/job && python train.py")
    await p.pull("/workspace/job/out", "./results")
# the pod is gone here — even if the body raised.
# No orphaned boxes, no surprise bills.
```

Like a hotel bellhop: books the room, waits until it's *actually* ready,
carries your luggage up, brings your bags back down, and checks out for you.

## Quick start

**1. Install:**

```bash
pip install "git+https://github.com/jonathanbostock/bellhop"            # RunPod + Lambda backends
pip install "bellhop-py[modal] @ git+https://github.com/jonathanbostock/bellhop"   # + Modal
pip install "bellhop-py[nebius] @ git+https://github.com/jonathanbostock/bellhop"  # + Nebius
```

(The distribution is `bellhop-py`; the import and CLI are plain `bellhop`.
RunPod and Lambda need only httpx and ship in the core install; Modal and
Nebius pull their provider SDKs, so they're extras.)

**2. Authenticate:**

- **RunPod**: `export RUNPOD_API_KEY=...`.
- **Lambda**: `export LAMBDA_API_KEY=...` (create one in the
  [Lambda console](https://cloud.lambda.ai)).
- **Modal**: `modal token new` (or `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`).
- **Nebius**: ambient SDK auth — `NEBIUS_IAM_TOKEN`, a service-account
  credentials file (`nebius iam auth-public-key generate`), or the `nebius`
  CLI config — plus `export NEBIUS_PROJECT_ID=project-e00...` (the project's
  region is where VMs land).

The SSH backends (RunPod, Lambda, Nebius) connect with your SSH keypair
(`~/.ssh/id_ed25519` by default): bellhop injects the public key into the box
automatically — RunPod via pod env, Lambda via its key registry, Nebius via
cloud-init.

**3. Run something:**

```python
import asyncio
from bellhop import run, RunSpec, PodConfig

res = asyncio.run(run(
    RunSpec(slug="demo", codebase="./mycode", run="python go.py"),
    PodConfig(gpu="A100"),
))
print(res.remote_exit, res.local_results)   # results pulled to ./experiments/demo/
```

or from the shell — same flags, any backend:

```bash
bellhop run --slug demo --codebase ./mycode --run "python go.py" --gpu A100
bellhop run --backend lambda --slug demo --codebase ./mycode --run "python go.py" --gpu H100
bellhop run --backend nebius --slug demo --codebase ./mycode --run "python go.py" --gpu H200 --gpu-count 8
```

That one call provisions a box, waits until it's genuinely reachable, uploads
`./mycode`, runs your command (logged to `results/run.log`), pulls the
`results/` directory back, and deletes the box — even if something fails
midway.

## Which mode do I want?

| You want to… | Use | Section |
|---|---|---|
| Run one job start-to-finish | `run()` / `bellhop run` | [One-shot jobs](#one-shot-jobs) |
| Keep a box alive for several steps | `pod()` / `sandbox()` / `instance()` / `vm()` | [Interactive boxes](#interactive-boxes) |
| Call a Python function remotely, get the return value | `box.call(fn, ...)` | [Remote function calls](#remote-function-calls) |
| Fan out a parameter sweep | `run_many()` | [Sweeps](#sweeps) |
| Train across multiple nodes (100B-scale) | `run_cluster()` | [Multi-node clusters](#multi-node-runpod-instant-clusters) |

Everything takes a `PodConfig` (RunPod), `ModalConfig` (Modal),
`LambdaConfig` (Lambda) or `NebiusConfig` (Nebius) — the pipelines are
identical on all of them; see
[Backends & configuration](#backends--configuration).

## One-shot jobs

`run()` is the full pipeline: provision → wait-functional → upload codebase
(local dir *or* git URL) → run `setup` then `run` (tee'd to
`results/run.log`) → pull the results dir → optional GCS upload → teardown →
`RunResult`.

```python
res = await run(
    RunSpec(
        slug="demo",
        codebase="./mycode",              # or "https://github.com/you/repo"
        setup="pip install -r requirements.txt",
        run="python train.py",
        results_subdir="results",         # what gets pulled back
        gcs_base="gs://bucket/experiments",  # optional; omit to skip GCS
    ),
    PodConfig(gpu="A100", container_disk_gb=100),
)
```

Good to know:

- `setup` and `run` execute under `set -e` — a failing setup aborts instead of
  running the job on a half-configured box.
- If the job step times out (opt-in `timeout=`), bellhop still pulls whatever
  landed in `results/` before raising, so the run stays debuggable.
- **`keep_on_failure=True`** (`--keep-on-failure`) leaves the box up when the
  job fails, times out, or the pull dies — a failed 14-hour run's checkpoint
  is worth more than the hourly rate. The box id is printed; it keeps billing
  until you tear it down. Boxes rejected before the job (provisioning,
  `host_check`) are still cleaned up. Programmatic version: set
  `box.keep = True` from anywhere ("keep if the checkpoint marker exists").
- Every failing step's **full** output lands in `<local_out>/failure.log`
  (`RemoteJobError.log_tail` is a 2000-char courtesy copy, not the record).
- GCS upload happens from *your* machine — cloud credentials never touch the
  box. Needs `gcloud` on your PATH.
- Jobs live under `/workspace/<slug>` on every backend — the SSH backends
  bootstrap a user-owned `/workspace` before yielding the box.
- `push()` excludes `.git`, `.venv`, caches — and **`.env*`**: a repo-root
  `.env` full of keys must never land on a (possibly community-hosted) box.
  Ship secrets via `RunSpec(env=...)` / `exec(env=...)`, which stay off the
  box's disk and argv.

### Big data on the box: pair with ferry

bellhop's own transfer paths (codebase up, `results/` back) are tar-over-ssh
through your machine — right for code and small results, wrong for weights.
Bulk bytes should move **box ↔ object store directly**; that's
[ferry](https://github.com/dtch1997/arsenal/tree/main/packages/ferry)'s job
(`pip install ferry-sync`), and the pairing is one line of env:

```python
import ferry

RunSpec(
    slug="train",
    codebase="./mycode",
    setup="pip install ferry-sync",
    run="python -c \"import ferry; ferry.ensure_rclone(); "
        "ferry.pull('gcs:my-bucket/weights/big-model/', '/workspace/weights/', transfers=16)\""
        " && python train.py",
    env=ferry.gcs_pod_env(),   # short-lived (~1h) GCS token as rclone env vars
)
```

`ferry.gcs_pod_env()` mints a ~1-hour access token from your local gcloud ADC
and exposes it to the box as an rclone remote named `gcs:` — no long-lived
credential ever touches the box, and access self-expires. (For jobs longer
than an hour that must *upload* at the end, use a scoped service-account key
via `RunSpec(env=...)` instead.)

## Interactive boxes

Keep one box alive and run many steps against it:

```python
from bellhop import pod, PodConfig

async with pod(PodConfig(gpu="RTX4090")) as p:
    await p.push("./code", "/workspace/job")
    await p.exec("cd /workspace/job && python train.py", env={"HF_TOKEN": tok})
    await p.exec("python eval.py")            # same pod, no re-provision
    await p.pull("/workspace/job/results", "./out")
    print(p.proxy_url(8000))                  # https://<id>-8000.proxy.runpod.net
# pod deleted here — even if the body raised (pass keep=True to leave it up)
```

Same shape on every backend — only the config and the provider noun change:

```python
async with sandbox(ModalConfig(gpu="A10G")) as b: ...       # Modal
async with instance(LambdaConfig(gpu="H100")) as i: ...     # Lambda
async with vm(NebiusConfig(gpu="H200", gpu_count=8)) as v: ...  # Nebius
```

"Ready" means more than the API saying running — sshd (RunPod) or cloud-init
(Nebius) typically lags, so bellhop polls a **readiness probe** before
yielding the box. The default (`SshProbe("true")`) suits ssh job boxes; for
servers use `HttpProbe(8000, "/health")` or `LogMarkerProbe("server up")`.
(Modal sandboxes are execable as soon as `create()` returns — no probe step.)

### Detached jobs: surviving your own death

`exec()` runs the remote command as a child of the ssh session — if the
*launcher* dies (laptop sleep, kill -9, network drop), sshd SIGHUPs the job
and a 21-hour training run dies with it. For anything long, start it
detached:

```python
async with pod(PodConfig(gpu="H100"), keep=True) as p:
    job = await p.exec_detached("cd /workspace/job && python train.py",
                                env={"HF_TOKEN": tok}, name="midtrain")
# ... laptop sleeps, process dies, whatever ...

# later, from a fresh process:
from bellhop import DetachedJob
async with pod(existing_or_new, keep=True) as p:   # or any live box handle
    job = DetachedJob(p, "midtrain")               # reattach by name
    print(await job.running(), await job.tail(20))
    res = await job.wait()                         # exit code + log tail
```

The job runs under `setsid` with its output in `/tmp/bellhop-jobs/<name>/`;
the handle is stateless, so any process that can reach the box can poll,
tail, or wait. Works on all three SSH backends.

### Picky about hosts: floors and rejection (RunPod)

RunPod schedules onto any host satisfying the GPU ask — which can mean a
512GB-RAM machine for your 1.9TB FSDP run, or the same defective host served
back seven times. Two levers:

```python
PodConfig(
    gpu="H200", gpu_count=8,
    min_memory_gb=1900, min_vcpu=96,     # host floors (GraphQL create path)
    host_check=my_check,                 # post-ready acceptance check
    host_check_retries=3,
)

async def my_check(pod):                 # raise to REJECT the host and reroll
    if pod.host in KNOWN_BAD_IPS:
        raise RuntimeError(f"blocklisted host {pod.host}")
    up = await pod.exec("dd if=/dev/zero bs=1M count=256 | gsutil cp - gs://bkt/probe && ...")
    if too_slow(up):                     # asymmetric faults are real: fine download,
        raise RuntimeError("upload 6MB/s")   # broken upload — only an upload probe sees it
```

A rejected host is torn down (even under `keep=True`) and provisioning
rerolls; exhaustion raises `ProvisionError` listing every rejection.

## Remote function calls

`call()` runs a local Python function on the box and returns its result as a
real Python object — no entrypoint script, no argv plumbing, no stdout
parsing:

```python
def train_probe(layer: int, lr: float) -> dict:
    import torch                      # imports resolve on the box
    ...
    return {"auc": auc, "loss": loss}

async with pod(PodConfig(gpu="RTX4090", pip=["torch", "scikit-learn"])) as p:
    metrics = await p.call(train_probe, layer=17, lr=1e-3)   # a real dict
```

Under the hood the function and arguments are `cloudpickle`d across; a remote
exception is re-raised locally as its original type with the remote traceback
attached. Async functions work too.

Rules of the road:

- **Python minor versions must match** between your machine and the box's
  interpreter (3.12 ↔ 3.11 fails). The first `call()` pre-flights this and
  raises `PreflightError` with the mismatch spelled out. Know your box's
  default `python3`: RunPod `pytorch-cuda` = 3.11, Lambda Stack (Ubuntu
  22.04) = **3.10**, Nebius `ubuntu24.04-*` = 3.12. The escape hatch is
  `call(..., python="python3.12")` after a `setup` that installs it.
- **Dependencies go on the config** (`pip=[...]` — installed post-readiness on
  the SSH backends, baked into the image on Modal).
- **Arguments/results travel by value** — fine for configs and metrics; ship
  large artifacts via GCS or a volume and pass paths.

## Sweeps

```python
from dataclasses import replace
from bellhop import run_many

base = RunSpec(slug="sweep", codebase="./code", run="python train.py")
specs = [replace(base, slug=f"lr{lr}", run=f"python train.py --lr {lr}")
         for lr in (1e-4, 3e-4, 1e-3)]
results = await run_many(specs, PodConfig(gpu="A100"), max_concurrency=4)
```

Each spec gets its own independent box; results/exceptions come back
positionally. On Lambda, concurrent launches are automatically paced to the
API's documented 1-launch-per-12s limit (process-wide), so a fan-out queues
briefly instead of erroring.

## Multi-node: RunPod Instant Clusters

For jobs one node can't hold, `run_cluster()` is the N-node sibling of
`run()`: same `RunSpec`, but the job runs on **every rank concurrently** with
the full distributed environment injected — `NODE_RANK`,
`PRIMARY_ADDR`/`MASTER_ADDR`, `PRIMARY_PORT`, `NUM_NODES`, `NUM_TRAINERS`,
`WORLD_SIZE`, and `NCCL_SOCKET_IFNAME=ens1` (the cluster's high-bandwidth
interconnect). Results are pulled from rank 0.

```python
from bellhop import ClusterConfig, RunSpec, run_cluster

spec = RunSpec(
    slug="train-100b", codebase="./code",
    run='torchrun --nnodes "$NUM_NODES" --node_rank "$NODE_RANK" '
        '--nproc_per_node "$NUM_TRAINERS" --rdzv_backend static '
        '--rdzv_endpoint "$PRIMARY_ADDR:$PRIMARY_PORT" train.py',
)
res = await run_cluster(spec, ClusterConfig(gpu="H200", nodes=4, gpu_count=8))
```

Lower-level, `async with cluster(config) as clu:` yields a `Cluster` whose
`nodes` are ordinary `Pod` channels indexed by rank, with `exec_all()` /
`push_all()` / `pull(rank=0)`.

Cluster-specific behavior to know about:

- **Rendezvous is bellhop-derived.** RunPod's docs promise `PRIMARY_ADDR`
  injection but it doesn't actually happen; bellhop reads each node's rank
  and overlay IP and injects the full env itself. Use
  `--rdzv_backend static` (RunPod doesn't support the dynamic `c10d`
  backend).
- **Pricing is auto-bid.** Clusters require a `deployCost` bid, and RunPod
  only reveals the minimum in a rejection error — bellhop bids that minimum,
  capped by `ClusterConfig(max_hourly_cost=...)` (whole-cluster $/hr).
- **First failing rank cancels the others** (with a static rendezvous the
  survivors would hang at the next collective) and raises `ClusterJobError`
  with per-rank results.
- **Clusters have no server-side TTL** — teardown is the context manager plus
  a client-side `max_lifetime` watchdog. Sweep up leaks with
  `bellhop clusters list` / `bellhop clusters gc --older-than-hours 24`.
- **RunPod's `deleteCluster` can orphan member pods** (cluster object gone,
  pods still billing — and the API keeps no record of which pods were
  members). bellhop's teardown deletes every member pod directly, verifies
  each is gone, and names survivors loudly. Membership is also persisted at
  create time to a local ledger (`~/.local/state/bellhop/clusters.jsonl`,
  override with `BELLHOP_CLUSTER_LEDGER`), so `clusters gc` reaps a dead
  cluster's leftover pods exactly, at any age, even after a crash.

Supported shapes: H100/H200/B200 (3200 Gbps interconnect) and A100
(1600 Gbps), 2–8 nodes. Full API contract and live-probe findings:
[`docs/design/instant-clusters.md`](docs/design/instant-clusters.md).

## Backends & configuration

All four backends implement the same contract (`exec` / `push` / `pull` /
`exists_remote` / `teardown`), so every pipeline above is provider-agnostic —
hand it the config for the provider you want.

Shared knobs, spelled the same everywhere:

| Knob | Meaning |
|---|---|
| `gpu=` | Canonical short name (`"A100"`, `"H100"`, `"L4"`, …); `None` = CPU box (RunPod/Modal/Nebius — Lambda is GPU-only). Verbatim provider ids also pass: a RunPod gpuTypeId, a Lambda `gpu_8x_h100_sxm5`, a Nebius `gpu-h100-sxm`. `"A100"` means 80GB everywhere (say `A100-40GB` on Lambda for the cheap cards). |
| `gpu_count=` | GPUs per box. Lambda: picks the `gpu_<n>x_*` type; Nebius: picks the preset (1 or 8 on SXM platforms). |
| `max_lifetime=` | Hard kill switch (`timedelta`) — **server-side** on RunPod/Modal, **in-process watchdog only** on Lambda/Nebius; see [Cleanup](#cleanup-how-boxes-die). |
| `pip=` | Extra packages (post-readiness install on the SSH backends; baked into the image on Modal). |
| `ready=`, `provision_timeout=`, `ready_timeout=` | The readiness probe and its windows (defaults are boot-time-aware per backend — an 8× Lambda box gets 30 min where a RunPod pod gets 5). |

What genuinely differs stays backend-specific:

| | RunPod (`PodConfig`, `pod()`) | Modal (`ModalConfig`, `sandbox()`) | Lambda (`LambdaConfig`, `instance()`) | Nebius (`NebiusConfig`, `vm()`) |
|---|---|---|---|---|
| Box | container pod, NAT'd ssh | Sandbox container | Ubuntu VM, `ubuntu@ip:22` | Ubuntu VM, `ubuntu@ip:22` |
| Auth | `RUNPOD_API_KEY` | Modal token | `LAMBDA_API_KEY` | Nebius IAM (+ `NEBIUS_PROJECT_ID`) |
| Placement | `cloud=` SECURE/COMMUNITY (+fallback) | `region=`, `cpu=`, `memory=` | `region=` (default: any with live capacity) | the project's region; `subnet_id=` |
| Image | `image=`/`image_preset=` docker refs | `modal.Image`, `pip=`/`apt=`, `secrets=`, `volumes=` | Lambda Stack (or `image=` family/id) | `image_family=` (default cuda drivers), `disk_gb=` |
| Server-side TTL | `stop_after`/`terminate_after` (GPU pods) | `timeout` (always enforced) | **none** | **none** |
| Capacity signal | error prose on create | n/a (serverless) | live `regions_with_capacity_available` per type | "Not enough resources" on create |
| Typical boot→ready | 30–60 s past RUNNING | instant | ~3–5 min (1×), 10–15 min (8×) | ~2–3 min + cloud-init |
| Default `python3` | image's (preset: 3.11) | image's | 3.10 (Lambda Stack 22.04) | 3.12 (ubuntu 24.04) |

Implementation notes: the RunPod and Lambda backends talk to their REST APIs
directly over `httpx` (RunPod GraphQL only where REST has no equivalent); the
Nebius control plane is gRPC-only, so that backend uses the official `nebius`
SDK (optional extra, lazily imported — like `modal`). Transfers are
tar-over-ssh (SSH backends) / tar-over-exec (Modal) — images only need `tar`.
Env vars passed to `exec(env=...)` are fed over stdin, never argv, so secrets
don't show up in the box's process list. Lambda API calls are paced to the
documented 1 req/s (and 1 launch/12 s) process-wide.

## Cleanup: how boxes die

Two independent layers:

| When | Handled by |
|------|------------|
| Normal exit, exception, Ctrl-C | the `async with` / pipeline `finally` — **always** tears down (unless `keep=True`) |
| Your process itself dies (kill -9, crash, reboot) | server-side safety timers — **where the provider has them** |

The context manager is the primary guarantee; the timers cover the one case
`finally` can't reach. Per backend:

- **RunPod** GPU pods get `stop_after=24h` (halt billing, disk persists) and
  `terminate_after=72h` (delete) unless you change them; enforcement is
  coarse (hours-scale backstop, not a precise kill switch), and CPU pods get
  no timer at all (bellhop warns).
- **Modal** *always* enforces a sandbox lifetime (its own default is 300 s) —
  bellhop warns if you leave `timeout` unset.
- **Lambda and Nebius have NO server-side TTL of any kind** — billing runs
  until the instance is terminated. `max_lifetime=` on those configs arms an
  *in-process* watchdog: it terminates the box when it fires (a mid-run kill
  surfaces as a clear "hit max_lifetime" error, not a mystery ssh failure),
  but it dies with your process, and `keep=True` disarms it (bellhop warns
  loudly in both cases). Before destroying the box, the watchdog runs a
  bounded **grace hook** — `run()` points it at an emergency results pull, so
  a run can't be lost to its own safety timer during the results phase (the
  cluster watchdog does the same for rank 0). The real backstop is the
  reaper — bellhop stamps every launch's name with `-t<epoch>` so leaks are
  findable:

  ```bash
  bellhop lambda list                                # everything, yours or not
  bellhop lambda gc --older-than-hours 24 --dry-run  # preview the reap
  bellhop nebius gc --older-than-hours 24            # stamped VMs only
  ```

  `gc` only ever touches bellhop-stamped boxes. Run it from cron if you run
  unattended sweeps.
- `exec()` (and the job step of `run()`) has **no client-side timeout by
  default**: long training runs go until they finish or the box dies. A
  *dead* connection is still detected promptly via SSH keepalives. Cap a
  specific command with `exec(..., timeout=7200)` /
  `RunSpec(..., timeout=7200)` / `--run-timeout-hours 2` — expiry raises
  `ExecTimeoutError`.

## Typed errors

Branch on failure mode instead of parsing strings: `PreflightError` (bad
config / missing key), `ProvisionError` (create failed — check
`is_capacity_error(e)` for stock-outs, on any provider), `PodNotReadyError`,
`RemoteJobError` (`.remote_exit`, `.log_tail`), `ClusterJobError` (per-rank
results), `ExecTimeoutError`, `RemoteCallError` (`.remote_traceback`),
`ResultsMissingError`, `GcsUploadError`. All subclass `BellhopError`.

## Development

```bash
pip install -e ".[dev]"
pytest                              # offline unit tests (no box, no cost)
RUNPOD_LIVE=1 pytest tests/integration_live.py -s     # billed RunPod e2e
MODAL_LIVE=1  pytest tests/integration_modal.py -s    # billed Modal e2e
LAMBDA_LIVE=1 pytest tests/integration_lambda.py -s   # billed Lambda e2e
NEBIUS_LIVE=1 pytest tests/integration_nebius.py -s   # billed Nebius e2e
```

Design notes for the Lambda/Nebius backends (API contracts, quirks, cleanup
model): [`docs/design/reliable-providers.md`](docs/design/reliable-providers.md).

## License

MIT
