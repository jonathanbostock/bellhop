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
  12 seconds. Polling sits at 10 s; consecutive launch attempts are spaced 12 s.
- **SSH keys**: `GET/POST /ssh-keys`. We look for a registered key matching
  the local public key (compare type+blob, ignore comment) and register it as
  `bellhop-<sha256[:12]>` if absent.
- **No TTL of any kind** (confirmed against the full OpenAPI spec): billing
  runs until terminate. `max_lifetime` is therefore a *client-side* watchdog
  (the `cluster.py` pattern) and the config warns that it does not survive
  host death.

GPU vocabulary: instance types are named `gpu_<count>x_<model>[_<variant>]`
(`gpu_1x_h100_pcie`, `gpu_8x_h100_sxm5`, `gpu_8x_a100_80gb_sxm4`, …).
`LAMBDA_GPU_ALIASES` maps the canonical short names to *variant* suffixes in
preference order; `gpu=` + `gpu_count=` expand to candidate type names, which
are then filtered against the live catalog (so a candidate that doesn't exist
at count N just drops out). `instance_type=` passes a verbatim name, like
`gpu_id=` on RunPod.

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
  passwordless sudo, plus a `runcmd` that creates `/workspace` owned by that
  user). Default user `ubuntu` to match Lambda muscle memory.
- **Lifecycle**: Create returns an Operation; await it, then poll
  `InstanceService.Get` until `RUNNING` and the public IP appears, then run
  the readiness probe (covers the cloud-init tail). States: CREATING /
  STARTING / RUNNING / STOPPING / STOPPED / DELETING / ERROR.
- **No server-side TTL** (nothing in the InstanceSpec proto): same
  client-side watchdog + warning as Lambda.
- Image families: `ubuntu24.04-cuda12` for GPU boxes, `ubuntu24.04-driverless`
  for CPU (the 22.04 families were deprecated 2026-06-01). Drivers are baked
  in, so no install delay after SSH.

## What's shared (`sshbox.py`)

`Pod.exec/push/pull/exists_remote/_ssh_raw` moved verbatim into an `SshBox`
base class; a backend supplies `_ssh_endpoint() -> (host, port)`, `ssh_user`,
and the private-key path. `Pod` keeps its NAT-mapped endpoint and public
surface unchanged; `LambdaInstance` / `NebiusInstance` are direct `ip:22`.
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
that terminates the box when it fires, and the config warns (once, at
provision time) that a `kill -9` of the host leaves the box running — sweep
with `bellhop lambda gc` / `bellhop nebius gc` (list+reap subcommands, the
`bellhop clusters` pattern).
