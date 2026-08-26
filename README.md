# bellhop

[![PyPI](https://img.shields.io/pypi/v/bellhop-py?color=blue)](https://pypi.org/project/bellhop-py/)
[![Python](https://img.shields.io/pypi/pyversions/bellhop-py)](https://pypi.org/project/bellhop-py/)
[![CI](https://github.com/dtch1997/arsenal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dtch1997/arsenal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Run your code on a disposable cloud GPU — provision, execute, bring results
back, tear down.** Async Python, two interchangeable backends:
[RunPod](https://runpod.io) pods and [Modal](https://modal.com) sandboxes.
Scales from one CPU box to a multi-node H200 cluster without changing shape.

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
pip install bellhop-py           # RunPod backend
pip install 'bellhop-py[modal]'  # + Modal backend
```

(The PyPI name is `bellhop-py`; the import and CLI are plain `bellhop`.)

**2. Authenticate:**

- **RunPod**: `export RUNPOD_API_KEY=...`. Connections use your SSH keypair
  (`~/.ssh/id_ed25519` by default — bellhop injects the public key into the
  pod automatically).
- **Modal**: `modal token new` (or `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`).

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

or from the shell:

```bash
bellhop run --slug demo --codebase ./mycode --run "python go.py" --gpu A100
```

That one call provisions a pod, waits until it's genuinely reachable, uploads
`./mycode`, runs your command (logged to `results/run.log`), pulls the
`results/` directory back, and deletes the pod — even if something fails
midway.

## Which mode do I want?

| You want to… | Use | Section |
|---|---|---|
| Run one job start-to-finish | `run()` / `bellhop run` | [One-shot jobs](#one-shot-jobs) |
| Keep a box alive for several steps | `pod()` / `sandbox()` | [Interactive boxes](#interactive-boxes) |
| Call a Python function remotely, get the return value | `box.call(fn, ...)` | [Remote function calls](#remote-function-calls) |
| Fan out a parameter sweep | `run_many()` | [Sweeps](#sweeps) |
| Train across multiple nodes (100B-scale) | `run_cluster()` | [Multi-node clusters](#multi-node-runpod-instant-clusters) |

Everything takes either a `PodConfig` (RunPod) or a `ModalConfig` (Modal) —
the pipelines are identical on both; see
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
- GCS upload happens from *your* machine — cloud credentials never touch the
  pod. Needs `gcloud` on your PATH.

### Big data on the pod: pair with ferry

bellhop's own transfer paths (codebase up, `results/` back) are tar-over-ssh
through your machine — right for code and small results, wrong for weights.
Bulk bytes should move **pod ↔ object store directly**; that's
[ferry](../ferry)'s job (`pip install ferry-sync`), and the pairing is one
line of env:

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
and exposes it to the pod as an rclone remote named `gcs:` — consistent with
the credo above: no long-lived credential ever touches the pod, and access
self-expires. (For jobs longer than an hour that must *upload* at the end,
use a scoped service-account key via `RunSpec(env=...)` instead.) Measured
throughput and resume semantics are in ferry's README stress section;
`packages/ferry/scripts/stress_pod.py` is the runnable proof of this exact
pairing.

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

Same shape on Modal: `async with sandbox(ModalConfig(gpu="A10G")) as b:`.

On RunPod, "ready" means more than the API saying RUNNING — sshd typically
lags by 30–60 s, so bellhop polls a **readiness probe** before yielding the
pod. The default (`SshProbe("true")`) suits ssh job pods; for servers use
`HttpProbe(8000, "/health")` or `LogMarkerProbe("server up")`.

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
  raises `PreflightError` with the mismatch spelled out.
- **Dependencies go on the config** (`pip=[...]` — installed post-readiness on
  RunPod, baked into the image on Modal). Pre-flight your pin set locally
  (`uv pip compile`) before burning pod-hours on a conflict.
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
positionally.

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
  a client-side `max_lifetime` watchdog. Sweep up leaks with:

  ```bash
  bellhop clusters list
  bellhop clusters gc --older-than-hours 24   # add --dry-run to preview
  ```

Supported shapes: H100/H200/B200 (3200 Gbps interconnect) and A100
(1600 Gbps), 2–8 nodes. Full API contract and live-probe findings:
[`docs/design/instant-clusters.md`](docs/design/instant-clusters.md).

## Backends & configuration

Both backends implement the same contract (`exec` / `push` / `pull` /
`exists_remote` / `teardown`), so every pipeline above is provider-agnostic —
hand it a `PodConfig` or a `ModalConfig`.

Shared knobs, spelled the same on both:

| Knob | Meaning |
|---|---|
| `gpu=` | Canonical short name (`"A100"`, `"H100"`, `"L4"`, …) or a full RunPod id; `None` = CPU box. On RunPod an alias expands to *all* matching SKUs (PCIe + SXM), which improves stock availability. |
| `max_lifetime=` | Hard server-side kill switch (`timedelta`) — see [Cleanup](#cleanup-how-boxes-die). |
| `image=` / `image_preset=` | The `pytorch-cuda` preset is pinned to the same torch 2.4.0 + CUDA 12.4 environment on both backends. |
| `pip=` | Extra packages (post-readiness install on RunPod; baked into the image on Modal). |

What genuinely differs stays backend-specific:

| | RunPod (`PodConfig`, `pod()`) | Modal (`ModalConfig`, `sandbox()`) |
|---|---|---|
| Readiness | SSH/probe wait | none — execable immediately |
| Extra TTL | `stop_after` (wall-clock compute halt) | `idle_timeout` (kill after inactivity) |
| Image extras | — | `apt=`, `modal.Image`, `secrets=`, `volumes=` |
| Placement | `cloud=` SECURE/COMMUNITY (+auto fallback on stock-out) | `region=`, `cpu=`, `memory=` |
| Auth | `RUNPOD_API_KEY` + SSH keypair | Modal token |

Implementation notes: the RunPod backend talks to the REST API directly over
`httpx` (GraphQL only where REST has no equivalent: native TTLs and Instant
Clusters); no `runpodctl`, no vendored SDK. Transfers are tar-over-ssh
(RunPod) / tar-over-exec (Modal) — images only need `tar`. Env vars passed to
`exec(env=...)` are fed over stdin, never argv, so secrets don't show up in
the box's process list.

## Cleanup: how boxes die

Two independent layers:

| When | Handled by |
|------|------------|
| Normal exit, exception, Ctrl-C | the `async with` / pipeline `finally` — **always** tears down (unless `keep=True`) |
| Your process itself dies (kill -9, crash, reboot) | server-side safety timers |

The context manager is the primary guarantee; the timers cover the one case
`finally` can't reach. On RunPod every GPU pod is created with
`stop_after=24h` (halt billing, disk persists) and `terminate_after=72h`
(delete) unless you change them; the backend-agnostic spelling
`max_lifetime=timedelta(...)` maps to `terminate_after` on RunPod and
`timeout` on Modal.

Caveats worth knowing:

- RunPod enforces its timers on a coarse schedule — treat them as an
  hours-scale backstop, not a precise kill switch. They also apply to GPU
  pods only (CPU pods rely on `finally`; bellhop warns when a CPU pod's TTL
  is dropped).
- Modal *always* enforces a sandbox lifetime (its own default is 300 s) —
  bellhop warns if you leave `timeout` unset.
- Instant Clusters have **no** server-side timers at all — see
  `bellhop clusters gc` above.
- `exec()` (and the job step of `run()`) has **no client-side timeout by
  default**: long training runs go until they finish or the box TTL fires. A
  *dead* connection is still detected promptly via SSH keepalives. Cap a
  specific command with `exec(..., timeout=7200)` /
  `RunSpec(..., timeout=7200)` / `--run-timeout-hours 2` — expiry raises
  `ExecTimeoutError`.

## Typed errors

Branch on failure mode instead of parsing strings: `PreflightError` (bad
config / missing key), `ProvisionError` (create failed — check
`is_capacity_error(e)` for stock-outs), `PodNotReadyError`, `RemoteJobError`
(`.remote_exit`, `.log_tail`), `ClusterJobError` (per-rank results),
`ExecTimeoutError`, `RemoteCallError` (`.remote_traceback`),
`ResultsMissingError`, `GcsUploadError`. All subclass `BellhopError`.

## Development

```bash
pip install -e ".[dev]"
pytest                              # offline unit tests (no box, no cost)
RUNPOD_LIVE=1 pytest tests/integration_live.py -s     # billed RunPod e2e
MODAL_LIVE=1  pytest tests/integration_modal.py -s    # billed Modal e2e
```

## License

MIT
