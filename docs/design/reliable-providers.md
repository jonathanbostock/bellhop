# Reliable-provider backends: Lambda Cloud + Nebius

Design notes for the two SSH-VM backends added in 0.9.0, and the API contracts
they encode. RunPod's spot-market pricing comes with capacity roulette and
opaque provisioning failures; Lambda Cloud and Nebius AI Cloud trade a little
of that price for boxes that reliably exist. Both are plain SSH-able VMs, so
they share the Pod's transport (tar-over-ssh, stdin-fed env) via the
`SshBox` base extracted from `pod.py`.

## The shape of each provider

|  | RunPod | Lambda Cloud | Nebius |
|---|---|---|---|
| API | REST + GraphQL | REST (`cloud.lambda.ai/api/v1`) | gRPC only (`nebius` SDK, optional extra) |
| Auth | `RUNPOD_API_KEY` | `LAMBDA_API_KEY` (Bearer) | IAM: `NEBIUS_IAM_TOKEN` / SA credentials file / CLI config |
| SSH user | `root` | `ubuntu` (fixed) | cloud-init-created (`ubuntu` by default) |
| SSH key delivery | `PUBLIC_KEY` env | registered key (auto-registered via `/ssh-keys`) | cloud-init `ssh_authorized_keys` |
| Port | NAT-mapped | direct :22 | direct :22 (dynamic one-to-one NAT IP) |
| Server-side TTL | GPU pods only | **none** | **none** |
| Capacity signal | create-error prose | `regions_with_capacity_available` per type | "Not enough resources" error |
| Boot to SSH | 30–60 s after RUNNING | ~3–5 min (1x), 10–15 min (8x) | ~2–3 min + cloud-init |

## Lambda Cloud (`lambda_box.py`)

REST over httpx, no SDK — same posture as the RunPod backend. Facts pinned
from the live OpenAPI spec (v1.10.0, fetched 2026-08-26):

- Base URL `https://cloud.lambda.ai/api/v1` (the old `cloud.lambdalabs.com`
  host still serves the same API but is marked deprecated). Success envelope
  `{"data": ...}`, error envelope `{"error": {"code", "message", ...}}` —
  branch on `code`, never on message text.
- **Launch**: `POST /instance-operations/launch` with `region_name`,
  `instance_type_name`, `ssh_key_names` (exactly one, must be pre-registered),
  optional `name` / `image` / `user_data` / `file_system_names`. One instance
  per call; returns `{"instance_ids": [id]}`.
- **States**: booting → active (also unhealthy / terminating / terminated /
  preempted). `ip` is not guaranteed present until late in boot — wait for
  `status == "active"` **and** `ip`, then run the readiness probe (sshd lags
  "active" slightly).
- **Capacity**: `GET /instance-types` returns every type with a live
  `regions_with_capacity_available` list — a real capacity signal RunPod
  doesn't offer. We pre-filter candidates against it, but still handle the
  launch-time race: out-of-stock launch is HTTP 400 with code
  `instance-operations/launch/insufficient-capacity`.
- **Rate limits** (documented): 1 request/second generally, and 1 launch per
  12 seconds. The pacer is *process-wide* (module-level monotonic timestamps —
  not an `asyncio.Lock`, which would bind the first event loop), because a
  `run_many()` fan-out launches from many `LambdaRest` clients concurrently
  and per-client pacing would deterministically trip the limit. 429s are
  retried with backoff on top. Polling sits at 10 s.
- **SSH keys**: `GET/POST /ssh-keys`. We look for a registered key matching
  the local public key (compare type+blob field-wise, ignore comment) and
  register it as `bellhop-<sha256[:12]>` if absent; a lost registration race
  (duplicate-name error) re-lists and matches instead of failing the launch.
- **No TTL of any kind** (confirmed against the full OpenAPI spec): billing
  runs until terminate. `max_lifetime` is therefore a *client-side* watchdog
  (the `cluster.py` pattern) and the config warns that it does not survive
  host death — see the cleanup story below.

GPU vocabulary: instance types are named `gpu_<count>x_<model>[_<variant>]`
(`gpu_1x_h100_pcie`, `gpu_8x_h100_sxm5`, `gpu_8x_a100_80gb_sxm4`, …).
`LAMBDA_GPU_ALIASES` maps the canonical short names to *variant* suffixes in
preference order; `gpu=` + `gpu_count=` expand to candidate type names, which
are then filtered against the live catalog (so a candidate that doesn't exist
at count N just drops out). `instance_type=` passes a verbatim name, like
`gpu_id=` on RunPod. Two deliberate choices: **"A100" means 80GB variants
only** (matching the RunPod alias — silently landing on 40GB cards OOMs jobs;
`A100-40GB` is the explicit spelling), and **a pinned `region=` is attempted
even when the catalog shows no capacity** — the capacity list is advisory and
launch is the source of truth (the TOCTOU race cuts both ways).

`/workspace` contract: the `ubuntu` user can't write `/`, so after readiness
the backend runs `sudo mkdir -p /workspace && sudo chown ubuntu /workspace` —
`run()`'s `/workspace/<slug>` layout then means the same thing on every
backend.

## Nebius (`nebius_box.py`)

The control plane is gRPC-only (no public REST), so this backend uses the
official `nebius` SDK — an optional extra (`pip install 'bellhop-py[nebius]'`),
lazily imported, exactly like `modal`. The SDK is async-first and reads
ambient credentials (env token, service-account file, or CLI config).

- **Resource model**: everything lives under a project
  (`project-e00…`, from `NebiusConfig.project_id` or `NEBIUS_PROJECT_ID`).
  Instance creation is a single `InstanceService.Create` with an inline
  *VM-managed* boot disk (`AttachedDiskSpec.managed_disk`, built from a public
  image family) and one network interface with a dynamic public IP
  (`PublicIPAddress()`); the subnet defaults to the project's first
  (`SubnetService.List`). Managed disks die with the VM in the same Delete —
  no separate disk cleanup, no orphaned-disk billing.
- **GPU vocabulary**: platform + preset, e.g. `gpu-h100-sxm` /
  `1gpu-16vcpu-200gb`. `NEBIUS_GPU_PLATFORMS` maps canonical names to
  platform ids; the preset is derived from `gpu_count` (SXM platforms come in
  1-GPU and 8-GPU presets). `platform=`/`preset=` pass verbatim.
- **SSH**: no default user on the images, root/admin are blocked — the login
  user is created by cloud-init (`users:` block with the local public key and
  passwordless sudo). Default user `ubuntu` to match Lambda muscle memory.
  `/workspace` is deliberately *not* made in cloud-init's `runcmd` — that
  runs in the final phase, often 30–90 s after sshd starts accepting the key,
  so the probe would pass before the dir existed. Both SSH backends instead
  run the same post-ready `sudo mkdir -p /workspace && sudo chown …` step.
- **Lifecycle**: Create returns an Operation whose `resource_id` is available
  immediately; we skip `op.wait()` (broken/hung in some SDK versions —
  pysdk#74) and go straight to polling `InstanceService.Get`. Fresh VMs
  report **STOPPED with `reconciling=True`** on the way to STARTING →
  RUNNING, so STOPPED alone is not failure — only ERROR/DELETING, or STOPPED
  once reconciliation ends. The public IP comes back **CIDR-suffixed**
  (`"1.2.3.4/32"`) and appears only once boot progresses. Every SDK call is
  bounded with `wait_for` — bad credentials otherwise hang the SDK's
  token-renew loop forever (dstack hit the same). The SDK is pinned `<0.7`
  (pre-1.0 proto churn) with an import-surface canary test.
- **No server-side TTL** (nothing in the InstanceSpec proto): same
  client-side watchdog + warning as Lambda. Teardown *retries* the delete —
  a delete racing the create tail can be refused, and a swallowed refusal on
  a TTL-less provider is a silent leak.
- Image families: `ubuntu24.04-cuda12` for GPU boxes, `ubuntu24.04-driverless`
  for CPU (the 22.04 families were deprecated 2026-06-01). Drivers are baked
  in, so no install delay after SSH.

## What's shared (`sshbox.py`)

`Pod.exec/push/pull/exists_remote/_ssh_raw` moved verbatim into an `SshBox`
base class; a backend supplies `_ssh_endpoint() -> (host, port)`, `ssh_user`,
and the private-key path. `Pod` keeps its NAT-mapped endpoint and public
surface unchanged; `LambdaInstance` / `NebiusVm` are direct `ip:22`.
The probes (`SshProbe`, `TcpProbe`, `LogMarkerProbe`) already speak this
interface (`_ssh_raw` / `host` / `mapped_port`), so they work on all three
SSH backends; `HttpProbe(via_proxy=True)` remains RunPod-specific.

## Cleanup story (updated)

| Backend | `finally` | Server-side backstop |
|---|---|---|
| RunPod GPU | yes | `stop_after`/`terminate_after` |
| RunPod CPU / clusters | yes | none (warned) |
| Modal | yes | `timeout` (always enforced) |
| Lambda | yes | **none** — client watchdog only (warned) |
| Nebius | yes | **none** — client watchdog only (warned) |

Both new configs accept `max_lifetime=`; it arms an in-process watchdog task
that terminates the box when it fires. The watchdog is honest about its
limits: it warns at provision time that a `kill -9` of the host leaves the
box running, it warns again if `keep=True` disarms it at exit, it retries a
failed teardown (then names the leaked box on stderr instead of
bare-suppressing), and it sets `_lifetime_expired` so a mid-run kill surfaces
as a clear "hit max_lifetime" error rather than a baffling ssh exit-255.

The *real* backstop is the reaper. Every launch stamps `-t<epoch>` onto the
box name (the Lambda Instance record has no created-at field; on Nebius one
mechanism for both providers beats two), and `bellhop lambda|nebius gc
--older-than-hours N [--dry-run]` terminates only bellhop-stamped boxes older
than the threshold — a hand-made instance, or even a bellhop-named one
without a parseable stamp, is never touched.

## Field reports: what a real training campaign broke

A 110B FSDP2 midtraining campaign (8×H200, ≥1.9TB host RAM) run on the RunPod
backend surfaced seven failure modes, ranked here by dollars lost; each maps
to a feature in this fork:

1. **Teardown-on-failure destroyed state.** One failed upload invocation →
   job exit 1 → teardown deleted the only copy of a 220GB checkpoint from a
   finished 14h run. → `run(keep_on_failure=True)` and the `box.keep`
   mid-session escape hatch (`keep=True` was all-or-nothing: it would also
   have orphaned every rejected pod).
2. **No host-quality filtering.** The same defective host was re-served 7+
   times; RunPod schedules onto any host meeting the GPU ask regardless of
   RAM. → `PodConfig(min_memory_gb=, min_vcpu=)` (GraphQL passthrough) and
   `host_check=` with raise-to-reroll semantics.
3. **Asymmetric network faults.** A host with fine download had a broken
   6MB/s *upload* — invisible to any download-based preflight, fatal to
   checkpoint publish. → `host_check` runs post-readiness, so upload probes
   are just `pod.exec(...)`.
4. **SIGHUP kills long jobs.** exec-over-ssh ties the job to the launcher's
   life. → `exec_detached()` / `DetachedJob` (setsid + on-box exit file,
   reattachable by name).
5. **`.env` shipped to community pods.** `push()` tarred the raw tree. →
   `.env*` in `TAR_EXCLUDES`.
6. **The safety timer destroyed its own results.** A 2-node run was killed by
   its own 3h `max_lifetime` *during the results pull*. → the watchdog grace
   hook (`on_lifetime_expiry`): salvage-pull before teardown, bounded at
   15 min so a wedged pull can't defeat the TTL.
7. **Diagnostics evaporated.** `log_tail` is 2000 chars in an exception
   nobody printed; RunPod's 401 body is literally `{"error":{}}`. →
   `failure.log` persists every failing step's full output; auth errors name
   the key, the header style, and the endpoint.
8. **`deleteCluster` orphans member pods.** The raw mutation can delete the
   cluster *object* while its pods keep running under default names (a
   "deleted" 2×4×H200 left two $22/hr pods billing 3.5h — and the runaway
   spend saturated the account's spend cap, which then masqueraded as a
   capacity drought for unrelated creates). → `_delete_cluster` treats the
   mutation as advisory: per-pod delete + three verify rounds + survivors
   named LOUDLY on stderr; and `bellhop clusters gc` gained an orphan sweep.
   Corollary for debugging: when creates suddenly fail "for capacity", check
   the account spend cap and sweep for orphans first.
9. **REST pod records carry no cluster linkage.** A live probe of
   `GET /v1/pods` (2026-08-26) shows a pod's complete key set has nothing
   cluster-ish in it — so once the cluster object is gone, RunPod cannot say
   which pods were members, and the first field-hit orphans were only
   identified by hand. → membership is persisted at create time (the
   `createCluster` response includes `pods { id }`) to a local JSONL ledger
   in XDG *state* (not cache — wiping a cache must never lose the only
   record of billing pods). The gc orphan sweep is ledger-driven and exact:
   dead cluster + listed pod still alive → reap at any age; an entry is
   dropped only once a round verifies every pod gone. Teardown narrows the
   entry to survivors (or drops it when clean), and gc adopts survivors of
   clusters it age-reaps, so even foreign clusters get covered once this
   machine has touched them. The old REST-linkage read stays as
   belt-and-braces — member pods *might* carry a field regular pods omit
   (unconfirmed; a live cluster is needed to check) — but nothing relies on
   it, and there are still no name/timestamp heuristics.
