"""The Nebius AI Cloud backend — a real VM behind a gRPC control plane.

Nebius (nebius.com) is the datacenter-grade end of the GPU spectrum: reserved
H100/H200/B200 capacity in real regions, per-project quotas, and VMs that are
plain Ubuntu machines. The control plane is gRPC-only — there is no public
REST — so this backend uses the official ``nebius`` SDK as an optional extra
(``pip install 'bellhop-py[nebius]'``), lazily imported exactly like
``modal``. The box itself is reached as ``ubuntu@ip:22``, so the transport is
:class:`bellhop.sshbox.SshBox` unchanged.

The provider-specific half, and the quirks it encodes (each learned from the
API docs or from SkyPilot/dstack's scar tissue):

- **One create call, no orphaned disks.** The boot disk is declared *inline*
  (``AttachedDiskSpec.managed_disk`` from a public image family), which Nebius
  deletes together with the VM — no separate disk cleanup, nothing to leak.
- **SSH is cloud-init.** The images have no default user and root/admin are
  blocked, so the login user (+ passwordless sudo + your public key) is
  created via ``cloud_init_user_data``. The ``/workspace`` contract dir is
  made *after* the readiness probe over SSH — not in cloud-init's ``runcmd``,
  which runs in the final phase and can lag sshd by minutes.
- **STOPPED is not failure.** Fresh instances report ``STOPPED`` with
  ``reconciling=True`` before ``STARTING`` → ``RUNNING``; only ``ERROR`` /
  ``DELETING`` (or ``STOPPED`` once reconciliation ends) are terminal.
- **IPs come back CIDR-suffixed** (``"1.2.3.4/32"``) and appear only once
  boot progresses; both are handled in ``host``.
- **Bad credentials hang the SDK** in its token-renew loop, so every call
  here is bounded by :func:`_rpc`'s ``wait_for`` and surfaces a typed error.
- **No server-side TTL exists** (nothing in the InstanceSpec proto), so
  ``max_lifetime=`` arms the same in-process watchdog as the Lambda backend,
  the name gets a ``-t<epoch>`` stamp, and ``bellhop nebius gc`` reaps what a
  dead host leaves behind.

Auth is ambient, in the SDK's order: ``credentials_file=`` (a service-account
credentials JSON from ``nebius iam auth-public-key generate``), else
``NEBIUS_IAM_TOKEN``, else the ``nebius`` CLI config. Resources live under a
project (``project-e00…``) — pass ``project_id=`` or export
``NEBIUS_PROJECT_ID``; the project's region decides where the VM lands.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from types import SimpleNamespace
from typing import AsyncIterator

from .errors import (
    BellhopError,
    PodNotReadyError,
    PreflightError,
    ProvisionError,
)
from .pod import _canon_gpu
from .probes import ReadyProbe, SshProbe
from .sshbox import (
    SshBox,
    ensure_workspace,
    install_pip,
    lifetime_watchdog,
    pubkey_text,
    resolve_ssh_key,
    stamp_epoch,
    stamp_name,
    wait_ready,
)

# Canonical GPU vocabulary -> Nebius platform ids. Presets (the vCPU/RAM
# shape) are a second axis: defaults below cover the SXM platforms' two
# shapes; anything else needs an explicit preset= (`nebius compute platform
# list` shows what a project can see).
NEBIUS_GPU_PLATFORMS: dict[str, str] = {
    "B200": "gpu-b200-sxm",
    "B300": "gpu-b300-sxm",
    "H100": "gpu-h100-sxm",
    "H200": "gpu-h200-sxm",
    "L40S": "gpu-l40s-a",
    "RTX6000": "gpu-rtx6000",
}
DEFAULT_CPU_PLATFORM = "cpu-d3"

_PLATFORM_LOOKUP = {_canon_gpu(k): v for k, v in NEBIUS_GPU_PLATFORMS.items()}

# Preset defaults where the shape is documented; verified for the H100/H200
# SXM platforms (identical presets) and the cpu-d3 default shape.
_PLATFORM_PRESETS: dict[str, dict[int, str]] = {
    "gpu-h100-sxm": {1: "1gpu-16vcpu-200gb", 8: "8gpu-128vcpu-1600gb"},
    "gpu-h200-sxm": {1: "1gpu-16vcpu-200gb", 8: "8gpu-128vcpu-1600gb"},
    "cpu-d3": {0: "4vcpu-16gb"},
}

DEFAULT_GPU_IMAGE_FAMILY = "ubuntu24.04-cuda12"     # drivers baked in
DEFAULT_CPU_IMAGE_FAMILY = "ubuntu24.04-driverless"

_RPC_TIMEOUT = 60.0


def _import_nebius() -> SimpleNamespace:
    """Import the SDK pieces lazily (optional dependency), as one namespace."""
    try:
        from nebius.aio.service_error import RequestError
        from nebius.api.nebius.common.v1 import ResourceMetadata
        from nebius.api.nebius.compute.v1 import (
            AttachedDiskSpec,
            CreateInstanceRequest,
            DeleteInstanceRequest,
            DiskSpec,
            GetInstanceRequest,
            InstanceServiceClient,
            InstanceSpec,
            InstanceStatus,
            IPAddress,
            ListInstancesRequest,
            ManagedDisk,
            NetworkInterfaceSpec,
            PublicIPAddress,
            ResourcesSpec,
            SourceImageFamily,
        )
        from nebius.api.nebius.vpc.v1 import ListSubnetsRequest, SubnetServiceClient
        from nebius.sdk import SDK
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise PreflightError(
            "the Nebius backend needs the `nebius` package — install it with "
            "`pip install 'bellhop-py[nebius]'`"
        ) from e
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith("_")})


async def _rpc(awaitable, what: str, timeout: float = _RPC_TIMEOUT):
    """Bound an SDK call: invalid credentials hang the SDK's token-renew loop
    forever (dstack works around the same bug), so every call gets a deadline
    and a typed error naming the likely cause."""
    try:
        return await asyncio.wait_for(awaitable, timeout)
    except asyncio.TimeoutError:
        raise BellhopError(
            f"nebius {what} timed out after {timeout:.0f}s — with valid credentials this "
            "is a network problem; with a bad NEBIUS_IAM_TOKEN / credentials file the "
            "SDK hangs in token renewal, so check auth first"
        ) from None


def _safe_name(name: str) -> str:
    """Nebius resource names: lowercase alnum + dashes."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", name.lower())).strip("-") or "bellhop"


@dataclass
class NebiusConfig:
    """Config for a Nebius VM. The Nebius-side peer of ``PodConfig``.

    A VM's location comes from its *project* (projects are regional), so
    there is no region field — pick the project in the region you want.
    """

    gpu: str | None = None                 # canonical short name ("H100", …); None = CPU box
    platform: str | None = None            # verbatim platform id, e.g. "gpu-h100-sxm" (peer of gpu_id)
    preset: str | None = None              # verbatim preset id, e.g. "8gpu-128vcpu-1600gb"
    gpu_count: int = 1                     # picks the preset on SXM platforms (1 or 8)
    project_id: str | None = None          # "project-e00…"; default $NEBIUS_PROJECT_ID
    subnet_id: str | None = None           # default: the project's first subnet
    image_family: str | None = None        # default cuda image (GPU) / driverless (CPU)
    disk_gb: int = 100                     # NETWORK_SSD boot disk; dies with the VM (min 40 for cuda images)
    # pip specs installed right after readiness, before the VM is yielded —
    # same semantics as PodConfig.pip.
    pip: list[str] = field(default_factory=list)
    name: str = "bellhop"                  # sanitized + stamped -t<epoch> (see gc_vms)
    # auth / connection
    ssh_key: str | None = None             # private key; default ~/.ssh/id_ed25519
    ssh_user: str = "ubuntu"               # created by cloud-init (root/admin are blocked by Nebius)
    credentials_file: str | None = None    # SA credentials JSON; default: ambient SDK auth
    # readiness. VM start is ~2-3 min plus a cloud-init tail before the user
    # exists; the waits exit early on success.
    ready: ReadyProbe = field(default_factory=lambda: SshProbe("true"))
    provision_timeout: timedelta = timedelta(seconds=600)
    ready_timeout: timedelta = timedelta(seconds=600)
    poll_interval: float = 10.0
    # CLIENT-SIDE watchdog only — Nebius has no server-side TTL at all, so
    # this dies with your process. `bellhop nebius gc` is the real backstop.
    max_lifetime: timedelta | None = None

    def resolve_platform(self) -> str:
        if self.gpu and self.platform:
            raise PreflightError("set gpu= (canonical name) or platform= (verbatim Nebius id), not both")
        if self.platform:
            return self.platform
        if not self.gpu:
            return DEFAULT_CPU_PLATFORM
        if self.gpu.startswith(("gpu-", "cpu-")):
            return self.gpu  # full Nebius platform id, pass verbatim
        hit = _PLATFORM_LOOKUP.get(_canon_gpu(self.gpu))
        if hit:
            return hit
        raise PreflightError(
            f"unknown gpu {self.gpu!r}; known aliases: {sorted(NEBIUS_GPU_PLATFORMS)} "
            "(a full Nebius platform id like 'gpu-h100-sxm' also works)"
        )

    def resolve_preset(self) -> str:
        if self.preset:
            return self.preset
        platform = self.resolve_platform()
        shapes = _PLATFORM_PRESETS.get(platform)
        key = self.gpu_count if platform.startswith("gpu-") else 0
        if not shapes or key not in shapes:
            raise PreflightError(
                f"no default preset for platform {platform!r} x{self.gpu_count}; set preset= "
                "(e.g. '8gpu-128vcpu-1600gb'; `nebius compute platform list` shows the shapes)"
            )
        return shapes[key]

    def resolve_image_family(self) -> str:
        if self.image_family:
            return self.image_family
        gpu = self.resolve_platform().startswith("gpu-")
        return DEFAULT_GPU_IMAGE_FAMILY if gpu else DEFAULT_CPU_IMAGE_FAMILY

    def resolve_project_id(self) -> str:
        project = self.project_id or os.environ.get("NEBIUS_PROJECT_ID")
        if not project:
            raise PreflightError(
                "Nebius needs a project id: pass project_id= or export NEBIUS_PROJECT_ID "
                "(`nebius iam project list` — the project's region is where the VM lands)"
            )
        return project

    def resolve_ssh_key(self) -> str:
        return resolve_ssh_key(self.ssh_key)

    def cloud_init(self) -> str:
        """The user-creating #cloud-config (images have no default user)."""
        if self.ssh_user in ("root", "admin"):
            raise PreflightError("Nebius reserves the root/admin usernames — pick another ssh_user")
        return (
            "#cloud-config\n"
            "users:\n"
            f"  - name: {self.ssh_user}\n"
            "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
            "    shell: /bin/bash\n"
            "    ssh_authorized_keys:\n"
            f"      - {pubkey_text(self.ssh_key)}\n"
        )

    def stamped_name(self) -> str:
        # The -t<epoch> stamp (bellhop- prefix forced) is gc_vms's
        # ownership+age marker; sanitized first for Nebius name rules.
        return stamp_name(_safe_name(self.name))


class NebiusVm(SshBox):
    """A live Nebius VM. Construct via :func:`vm` (the async context manager).

    Transport comes from :class:`SshBox`; the box is ``ubuntu@<public ip>:22``
    directly (dynamic one-to-one NAT IP), so ``mapped_port`` is the identity
    and the standard probes work unchanged.
    """

    _noun = "vm"

    def __init__(self, sdk, nb: SimpleNamespace, vm_id: str, config: NebiusConfig):
        self._sdk = sdk
        self._nb = nb
        self.id = vm_id
        self.config = config
        self._inst = None                    # last-fetched Instance proto
        self._ssh_key = config.resolve_ssh_key()
        self._lifetime_expired = False

    # ---- connection info ---------------------------------------------------
    @property
    def ssh_user(self) -> str:
        return self.config.ssh_user

    @property
    def host(self) -> str | None:
        status = getattr(self._inst, "status", None)
        for nic in getattr(status, "network_interfaces", None) or []:
            addr = getattr(getattr(nic, "public_ip_address", None), "address", None)
            if addr:
                return addr.split("/")[0]  # comes back CIDR-suffixed ("1.2.3.4/32")
        return None

    def mapped_port(self, container_port: int = 22) -> int:
        return container_port  # direct networking, no NAT mapping

    @property
    def state(self) -> str:
        status = getattr(self._inst, "status", None)
        raw = getattr(status, "state", 0)
        try:
            return self._nb.InstanceStatus.InstanceState(int(raw)).name
        except ValueError:
            return f"UNKNOWN({raw})"

    @property
    def reconciling(self) -> bool:
        return bool(getattr(getattr(self._inst, "status", None), "reconciling", False))

    # ---- lifecycle ---------------------------------------------------------
    async def refresh(self):
        nb = self._nb
        self._inst = await _rpc(
            nb.InstanceServiceClient(self._sdk).get(nb.GetInstanceRequest(id=self.id)),
            "get_instance",
        )
        return self._inst

    async def _wait_provision(self) -> None:
        deadline = time.monotonic() + self.config.provision_timeout.total_seconds()
        while True:
            await self.refresh()
            state = self.state
            if state == "ERROR" or state == "DELETING":
                raise ProvisionError(f"vm {self.id} entered terminal state {state}")
            if state == "STOPPED" and not self.reconciling:
                # Fresh VMs *start* in STOPPED with reconciling=True on the way
                # to STARTING; STOPPED with reconciling over means it's not coming.
                raise ProvisionError(f"vm {self.id} settled in STOPPED — create failed upstream")
            if state == "RUNNING" and self.host:
                return
            if time.monotonic() >= deadline:
                raise PodNotReadyError(
                    f"vm {self.id} not RUNNING+addressed within "
                    f"{self.config.provision_timeout.total_seconds():.0f}s (state={state})"
                )
            await asyncio.sleep(self.config.poll_interval)

    async def _wait_ready(self) -> None:
        await wait_ready(self, self.config.ready, self.config.ready_timeout,
                         self.config.poll_interval)

    async def teardown(self) -> None:
        """Delete the VM (its managed boot disk goes with it, same operation).

        Deletion is retried: a delete racing the tail of the create operation
        can be refused (FAILED_PRECONDITION), and on a TTL-less provider a
        swallowed refusal = a silent leak. The last failure names the VM so
        it can be reaped by hand / `bellhop nebius gc`.
        """
        nb = self._nb
        last: Exception | None = None
        for delay in (0, 10, 30):
            if delay:
                await asyncio.sleep(delay)
            try:
                await _rpc(
                    nb.InstanceServiceClient(self._sdk).delete(nb.DeleteInstanceRequest(id=self.id)),
                    "delete_instance",
                )
                return
            except nb.RequestError as e:
                if "not_found" in str(e).lower().replace(" ", "_"):
                    return  # already gone; idempotent teardown
                last = e
            except Exception as e:
                last = e
        print(f"bellhop: could not delete nebius vm {self.id} ({last}) — it is still "
              "billing; reap with `bellhop nebius gc`", file=sys.stderr, flush=True)
        raise last

    # ---- exec / transfer: inherited from SshBox ------------------------------
    def _ssh_endpoint(self) -> tuple[str, int]:
        host = self.host
        if not host:
            raise PodNotReadyError("vm has no public ip yet")
        return host, 22


def _make_sdk(nb: SimpleNamespace, config: NebiusConfig):
    if config.credentials_file:
        return nb.SDK(credentials_file_name=config.credentials_file)
    return nb.SDK()  # ambient: NEBIUS_IAM_TOKEN, else the CLI config


async def _pick_subnet(nb: SimpleNamespace, sdk, project_id: str) -> str:
    resp = await _rpc(
        nb.SubnetServiceClient(sdk).list(nb.ListSubnetsRequest(parent_id=project_id)),
        "list_subnets",
    )
    subnets = getattr(resp, "items", None) or []
    if not subnets:
        raise PreflightError(
            f"project {project_id} has no subnets — every Nebius project normally has a "
            "default one; check the project id (and `nebius vpc subnet list`)"
        )
    return subnets[0].metadata.id


def _create_request(nb: SimpleNamespace, config: NebiusConfig,
                    project_id: str, subnet_id: str):
    """Assemble the CreateInstanceRequest. Pure (no network) so it's unit-testable."""
    name = config.stamped_name()
    return nb.CreateInstanceRequest(
        metadata=nb.ResourceMetadata(parent_id=project_id, name=name),
        spec=nb.InstanceSpec(
            resources=nb.ResourcesSpec(platform=config.resolve_platform(),
                                       preset=config.resolve_preset()),
            boot_disk=nb.AttachedDiskSpec(
                attach_mode=nb.AttachedDiskSpec.AttachMode.READ_WRITE,
                # VM-managed: Nebius deletes it together with the VM.
                managed_disk=nb.ManagedDisk(
                    name=f"{name}-boot",
                    spec=nb.DiskSpec(
                        type=nb.DiskSpec.DiskType.NETWORK_SSD,
                        size_gibibytes=config.disk_gb,
                        source_image_family=nb.SourceImageFamily(
                            image_family=config.resolve_image_family()),
                    ),
                ),
            ),
            network_interfaces=[nb.NetworkInterfaceSpec(
                name="eth0",
                subnet_id=subnet_id,
                ip_address=nb.IPAddress(),
                public_ip_address=nb.PublicIPAddress(),  # dynamic one-to-one NAT
            )],
            cloud_init_user_data=config.cloud_init(),
        ),
    )


@contextlib.asynccontextmanager
async def vm(config: NebiusConfig, *, keep: bool = False) -> AsyncIterator[NebiusVm]:
    """Create a Nebius VM, wait until it's functional, yield it, delete it.

    On any exception (including a readiness timeout) the VM is still deleted —
    boot disk included — unless ``keep=True``. There is NO server-side TTL on
    Nebius: ``keep=True`` (or a killed host) leaves the VM billing until you
    delete it — ``bellhop nebius gc`` reaps stamped leftovers.
    """
    nb = _import_nebius()
    project_id = config.resolve_project_id()   # preflight before spending money
    config.resolve_preset()
    config.resolve_ssh_key()
    if config.max_lifetime:
        warnings.warn(
            "Nebius has no server-side TTL: max_lifetime is enforced by an "
            "in-process watchdog only — if this process dies, nothing deletes "
            "the VM (sweep leaks with `bellhop nebius gc`)",
            stacklevel=3,
        )
    sdk = _make_sdk(nb, config)
    try:
        subnet_id = config.subnet_id or await _pick_subnet(nb, sdk, project_id)
        req = _create_request(nb, config, project_id, subnet_id)
        try:
            op = await _rpc(nb.InstanceServiceClient(sdk).create(req), "create_instance")
        except (nb.RequestError, BellhopError) as e:
            raise e if isinstance(e, BellhopError) else ProvisionError(
                f"nebius instance create failed: {e}") from e
        vm_id = op.resource_id
        if not vm_id:
            raise ProvisionError(f"could not parse vm id from create operation: {op}")
        box = NebiusVm(sdk, nb, vm_id, config)
        # Armed the moment the id exists — billing starts at create, and the
        # bootstrap execs have no client timeout (see the Lambda backend for
        # the full rationale). max_lifetime is measured from create.
        watchdog: asyncio.Task | None = None
        if config.max_lifetime:
            watchdog = asyncio.create_task(lifetime_watchdog(box, config.max_lifetime))
        try:
            await box._wait_provision()
            await box._wait_ready()
            await ensure_workspace(box)
            if config.pip:
                await install_pip(box, config.pip)
            try:
                yield box
            except BaseException as e:
                if box._lifetime_expired and not isinstance(e, asyncio.CancelledError):
                    # The watchdog killed the box mid-run; say so instead of
                    # letting the ssh exit-255 masquerade as a job failure.
                    # (External cancellation stays a CancelledError.)
                    raise BellhopError(
                        f"vm {box.id} hit max_lifetime "
                        f"{config.max_lifetime} mid-run and was deleted"
                    ) from e
                raise
            if box._lifetime_expired:
                # Post-kill execs return rc=255 rather than raising — leave a
                # loud trace even when the body "completed".
                warnings.warn(
                    f"vm {box.id} hit max_lifetime {config.max_lifetime} "
                    "during the session — later commands ran against a dead box",
                    stacklevel=3,
                )
        finally:
            if watchdog:
                if box._lifetime_expired:
                    # cancel() here could abort the delete mid-flight (or its
                    # retry sleeps) and leak the box — let the bounded
                    # watchdog finish instead.
                    with contextlib.suppress(Exception):
                        await watchdog
                else:
                    watchdog.cancel()
            if (keep or box.keep) and config.max_lifetime and not box._lifetime_expired:
                warnings.warn(
                    f"keep=True disarms the max_lifetime watchdog: vm {box.id} "
                    "now has NO lifetime bound (reap with `bellhop nebius gc`)",
                    stacklevel=3,
                )
            if not (keep or box.keep):
                with contextlib.suppress(Exception):
                    await box.teardown()
    finally:
        with contextlib.suppress(Exception):
            await sdk.close()


# ---- leak reaping (no server-side TTL => leaks are on us to find) -----------

async def list_vms(project_id: str | None = None,
                   credentials_file: str | None = None) -> list[dict]:
    """The project's VMs as plain dicts (bellhop-launched or not)."""
    nb = _import_nebius()
    cfg = NebiusConfig(project_id=project_id, credentials_file=credentials_file)
    sdk = _make_sdk(nb, cfg)
    try:
        resp = await _rpc(
            nb.InstanceServiceClient(sdk).list(
                nb.ListInstancesRequest(parent_id=cfg.resolve_project_id(), page_size=999)),
            "list_instances",
        )
        out = []
        for inst in getattr(resp, "items", None) or []:
            state = getattr(getattr(inst, "status", None), "state", 0)
            try:
                state = nb.InstanceStatus.InstanceState(int(state)).name
            except ValueError:
                state = str(state)
            out.append({"id": inst.metadata.id, "name": inst.metadata.name, "state": state})
        return out
    finally:
        with contextlib.suppress(Exception):
            await sdk.close()


async def gc_vms(older_than: timedelta, *, dry_run: bool = False,
                 project_id: str | None = None,
                 credentials_file: str | None = None) -> list[dict]:
    """Delete bellhop-stamped VMs older than ``older_than``.

    Only VMs whose *name* carries the ``bellhop…-t<epoch>`` stamp are
    candidates — anything else in the project is never touched. Returns the
    reaped (or would-reap) VMs with an ``age_hours`` field.
    """
    nb = _import_nebius()
    cfg = NebiusConfig(project_id=project_id, credentials_file=credentials_file)
    now = time.time()
    reaped: list[dict] = []
    sdk = _make_sdk(nb, cfg)
    try:
        resp = await _rpc(
            nb.InstanceServiceClient(sdk).list(
                nb.ListInstancesRequest(parent_id=cfg.resolve_project_id(), page_size=999)),
            "list_instances",
        )
        for inst in getattr(resp, "items", None) or []:
            epoch = stamp_epoch(inst.metadata.name)
            state = getattr(getattr(inst, "status", None), "state", 0)
            if epoch is None or int(state) == int(nb.InstanceStatus.InstanceState.DELETING):
                continue
            age = now - epoch
            if age < older_than.total_seconds():
                continue
            entry = {"id": inst.metadata.id, "name": inst.metadata.name,
                     "age_hours": round(age / 3600, 1)}
            reaped.append(entry)
            if not dry_run:
                await _rpc(
                    nb.InstanceServiceClient(sdk).delete(
                        nb.DeleteInstanceRequest(id=inst.metadata.id)),
                    "delete_instance",
                )
        return reaped
    finally:
        with contextlib.suppress(Exception):
            await sdk.close()
