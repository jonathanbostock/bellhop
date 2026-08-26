"""The Pod resource: provision, wait-until-functional, exec / push / pull.

This is the composable layer the bash driver can't offer — keep a pod alive and
run many steps against it. ``run()`` (see run.py) is just this plus GCS upload.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Awaitable, Callable, Literal

from .errors import PodNotReadyError, PreflightError, ProvisionError
from .graphql import RunpodGraphQL
from .probes import ReadyProbe, SshProbe
from .rest import RunpodRest
from .sshbox import (
    SSH_OPTS,
    SshBox,
    install_pip,
    pubkey_text,
    resolve_ssh_key,
    wait_ready,
)

__all__ = ["GPU_ALIASES", "IMAGE_PRESETS", "Pod", "PodConfig", "SSH_OPTS", "pod"]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Image presets — kept inline so the library is standalone (no jarvis catalog).
# "pytorch-cuda" is torch 2.4.0 + CUDA 12.4, kept in lockstep with the Modal
# preset of the same name (modal_box._preset_image) so the key means the same
# environment on either backend.
IMAGE_PRESETS = {
    "cpu-base": "runpod/base:1.0.2-ubuntu2204",
    "pytorch-cuda": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "pytorch-latest": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
}

# Canonical GPU vocabulary — the Modal-style short names, expanded to the
# RunPod gpuTypeIds that satisfy them. REST's ``gpuTypeIds`` takes the whole
# candidate list (any match wins), so an alias also improves stock availability
# over naming one exact SKU.
GPU_ALIASES: dict[str, list[str]] = {
    "A40": ["NVIDIA A40"],
    "A100": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    "A100-80GB": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    "A6000": ["NVIDIA RTX A6000"],
    "B200": ["NVIDIA B200"],
    "H100": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"],
    "H200": ["NVIDIA H200"],
    "L4": ["NVIDIA L4"],
    "L40": ["NVIDIA L40"],
    "L40S": ["NVIDIA L40S"],
    "RTX4090": ["NVIDIA GeForce RTX 4090"],
    "RTX5090": ["NVIDIA GeForce RTX 5090"],
}


def _canon_gpu(name: str) -> str:
    return name.upper().replace(" ", "").replace("_", "").replace("-", "")


_ALIAS_LOOKUP = {_canon_gpu(k): v for k, v in GPU_ALIASES.items()}
DEFAULT_GPU_IMAGE = IMAGE_PRESETS["pytorch-cuda"]
DEFAULT_CPU_IMAGE = IMAGE_PRESETS["cpu-base"]


@dataclass
class PodConfig:
    compute: Literal["cpu", "gpu"] | None = None   # derived from gpu/gpu_id when omitted
    gpu: str | None = None                 # canonical short name ("A100", "H100", …) or full RunPod gpuTypeId; None = CPU
    gpu_id: str | None = None              # verbatim RunPod gpuTypeId (legacy spelling of gpu=)
    gpu_count: int = 1
    image: str | None = None               # free-form; wins over preset
    image_preset: str | None = None        # key into IMAGE_PRESETS
    container_disk_gb: int = 20
    volume_gb: int | None = None           # network-volume persistence
    volume_mount_path: str = "/workspace"
    cloud: Literal["SECURE", "COMMUNITY"] = "COMMUNITY"
    cloud_fallback: bool = True            # COMMUNITY out-of-stock -> retry SECURE
    ports: list[str] = field(default_factory=lambda: ["22/tcp"])
    env: dict[str, str] = field(default_factory=dict)
    # pip specs installed right after readiness, before the pod is yielded —
    # the RunPod peer of ModalConfig.pip (there it's baked into the image;
    # here it's a post-ready `python3 -m pip install`). Pre-flight conflicts
    # locally (`uv pip compile`) before burning pod-hours on a bad pin set.
    pip: list[str] = field(default_factory=list)
    # Host CUDA-driver filter. RunPod schedules onto any host meeting the
    # IMAGE's CUDA floor, which can be far older than what your wheels need
    # (a cu13-linked vllm wheel dies on a 12.9-driver host with "driver too
    # old"). List the acceptable host CUDA versions, e.g. ["13.0", "13.1"].
    cuda_versions: list[str] | None = None
    # Container start command override (single shell command string). Lets
    # non-RunPod images (which don't consume PUBLIC_KEY or start sshd) work as
    # ssh job pods — e.g. bootstrap sshd and block:
    #   "apt-get update && apt-get install -y openssh-server && mkdir -p /run/sshd ~/.ssh
    #    && echo \"$PUBLIC_KEY\" > ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    #    && /usr/sbin/sshd -D"
    # The command IS the container's main process: it must block (sshd -D,
    # sleep infinity) or the pod exits and crashloops.
    docker_start_cmd: str | None = None
    name: str = "bellhop"
    # auth / connection
    ssh_key: str | None = None             # private key; default ~/.ssh/id_ed25519
    ssh_user: str = "root"
    # Host-quality floors (GraphQL create path, GPU pods only). RunPod
    # schedules onto any host satisfying the GPU ask; a 110B FSDP run that
    # needs 1.9TB host RAM can land on a 512GB machine without these.
    min_memory_gb: int | None = None       # GraphQL minMemoryInGb
    min_vcpu: int | None = None            # GraphQL minVcpuCount
    # Post-readiness host acceptance check: async callable(pod) -> None that
    # RAISES to reject this host — the pod is torn down and provisioning
    # rerolls (up to host_check_retries). The pod is functional when called,
    # so checks can exec: IP blocklists, upload-bandwidth probes (a host with
    # fine download and a broken 6MB/s upload passes every download test and
    # then fails your checkpoint publish), nvme smoke tests, ...
    host_check: "Callable[[Pod], Awaitable[None]] | None" = None
    host_check_retries: int = 3
    # readiness. Defaults resolve in __post_init__: 300s/420s normally, but
    # 1200s each when docker_start_cmd is set — a bootstrap that apt-installs
    # its way to sshd routinely needs 5-15 min before the pod is reachable,
    # and the wait loops exit early on success so a generous cap is free.
    ready: ReadyProbe = field(default_factory=lambda: SshProbe("true"))
    provision_timeout: timedelta | None = None
    ready_timeout: timedelta | None = None
    poll_interval: float = 8.0
    # native server-side safety timers (GraphQL only; survive host death).
    # stop = halt compute (disk persists); terminate = delete (all billing stops).
    stop_after: timedelta | None = timedelta(hours=24)
    terminate_after: timedelta | None = timedelta(hours=72)
    # unified spelling of the hard kill, same name as ModalConfig.max_lifetime;
    # wins over BOTH timers when set: terminate_after takes its value and
    # stop_after is cleared (want a separate early-stop? set the two timers
    # directly instead of max_lifetime).
    max_lifetime: timedelta | None = None

    def __post_init__(self):
        slow_boot = bool(self.docker_start_cmd)
        if self.provision_timeout is None:
            self.provision_timeout = timedelta(seconds=1200 if slow_boot else 300)
        if self.ready_timeout is None:
            self.ready_timeout = timedelta(seconds=1200 if slow_boot else 420)
        if self.max_lifetime is not None:
            self.terminate_after = self.max_lifetime
            # the default 24h stop timer would halt a longer job early
            self.stop_after = None

    @property
    def resolved_compute(self) -> str:
        if self.compute:
            return self.compute
        return "gpu" if (self.gpu or self.gpu_id) else "cpu"

    def resolve_gpu_ids(self) -> list[str]:
        """The RunPod gpuTypeIds this config asks for, in preference order."""
        if self.gpu and self.gpu_id:
            raise PreflightError("set gpu= (canonical name) or gpu_id= (verbatim RunPod id), not both")
        if self.gpu_id:
            return [self.gpu_id]
        if not self.gpu:
            raise PreflightError("gpu required when compute='gpu' (e.g. gpu='A100')")
        hit = _ALIAS_LOOKUP.get(_canon_gpu(self.gpu))
        if hit:
            return list(hit)
        if self.gpu.upper().startswith(("NVIDIA", "AMD", "TESLA")):
            return [self.gpu]  # full RunPod gpuTypeId, pass verbatim
        raise PreflightError(
            f"unknown gpu {self.gpu!r}; known aliases: {sorted(GPU_ALIASES)} "
            "(a full RunPod gpuTypeId like 'NVIDIA GeForce RTX 4090' also works)"
        )

    def resolve_image(self) -> str:
        if self.image:
            return self.image
        if self.image_preset:
            try:
                return IMAGE_PRESETS[self.image_preset]
            except KeyError:
                raise PreflightError(
                    f"unknown image_preset {self.image_preset!r} (have {list(IMAGE_PRESETS)})"
                )
        return DEFAULT_GPU_IMAGE if self.resolved_compute == "gpu" else DEFAULT_CPU_IMAGE

    def resolve_ssh_key(self) -> str:
        return resolve_ssh_key(self.ssh_key)

    def pubkey_text(self) -> str:
        return pubkey_text(self.ssh_key)

    def to_create_body(self) -> dict:
        env = dict(self.env)
        env.setdefault("PUBLIC_KEY", self.pubkey_text())  # RunPod injects into authorized_keys
        body: dict = {
            "name": self.name,
            "imageName": self.resolve_image(),
            "cloudType": self.cloud,
            "containerDiskInGb": self.container_disk_gb,
            "ports": self.ports,
            "env": env,
        }
        if self.docker_start_cmd:
            body["dockerStartCmd"] = ["bash", "-c", self.docker_start_cmd]
        if self.cuda_versions:
            body["allowedCudaVersions"] = list(self.cuda_versions)
        if self.resolved_compute == "gpu":
            body["gpuTypeIds"] = self.resolve_gpu_ids()
            body["gpuCount"] = self.gpu_count
        else:
            body["computeType"] = "CPU"
        if self.volume_gb:
            body["volumeInGb"] = self.volume_gb
            body["volumeMountPath"] = self.volume_mount_path
        return body

    def has_ttl(self) -> bool:
        return bool(self.stop_after or self.terminate_after)

    def has_host_floor(self) -> bool:
        return bool(self.min_memory_gb or self.min_vcpu)

    def needs_graphql(self) -> bool:
        """REST v1 has neither TTL nor host-floor fields; either routes create
        through GraphQL (which handles both, TTL or not)."""
        return self.has_ttl() or self.has_host_floor()

    def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
        """Input for podFindAndDeployOnDemand — the only create path with TTL.

        Note the GraphQL shape differs from REST: gpuTypeId is singular (pass
        ``gpu_type_id`` to pick one candidate; default is the first), ports is
        a comma-joined string, and env is a list of {key, value} objects.
        """
        if self.resolved_compute != "gpu":
            raise PreflightError("native TTL (stop_after/terminate_after) requires a GPU box (set gpu= or gpu_id=)")
        env = dict(self.env)
        env.setdefault("PUBLIC_KEY", self.pubkey_text())
        inp: dict = {
            "cloudType": self.cloud,
            "name": self.name,
            "imageName": self.resolve_image(),
            "gpuTypeId": gpu_type_id or self.resolve_gpu_ids()[0],
            "gpuCount": self.gpu_count,
            "containerDiskInGb": self.container_disk_gb,
            "ports": ",".join(self.ports),
            "env": [{"key": k, "value": v} for k, v in env.items()],
        }
        if self.docker_start_cmd:
            # GraphQL's dockerArgs is the single-string spelling of REST's dockerStartCmd
            inp["dockerArgs"] = f"bash -c {shlex.quote(self.docker_start_cmd)}"
        if self.cuda_versions:
            inp["allowedCudaVersions"] = list(self.cuda_versions)
        if self.min_memory_gb:
            inp["minMemoryInGb"] = self.min_memory_gb
        if self.min_vcpu:
            inp["minVcpuCount"] = self.min_vcpu
        if self.volume_gb:
            inp["volumeInGb"] = self.volume_gb
            inp["volumeMountPath"] = self.volume_mount_path
        now = datetime.now(timezone.utc)
        if self.stop_after:
            inp["stopAfter"] = _iso(now + self.stop_after)
        if self.terminate_after:
            inp["terminateAfter"] = _iso(now + self.terminate_after)
        return inp


class Pod(SshBox):
    """A live pod. Construct via :func:`pod` (the async context manager).

    Transport (exec / push / pull / probes) comes from :class:`SshBox`; this
    class adds the RunPod-specific halves: REST lifecycle and the NAT-mapped
    SSH endpoint.
    """

    _noun = "pod"

    def __init__(self, rest: RunpodRest, pod_id: str, config: PodConfig):
        self._rest = rest
        self.id = pod_id
        self.config = config
        self._meta: dict = {}
        self._ssh_key = config.resolve_ssh_key()

    @property
    def ssh_user(self) -> str:
        return self.config.ssh_user

    # ---- connection info ---------------------------------------------------
    @property
    def host(self) -> str | None:
        return self._meta.get("publicIp")

    def mapped_port(self, container_port: int = 22) -> int | None:
        return (self._meta.get("portMappings") or {}).get(str(container_port))

    @property
    def status(self) -> str:
        return (self._meta.get("desiredStatus") or "UNKNOWN").upper()

    def proxy_url(self, container_port: int) -> str:
        return f"https://{self.id}-{container_port}.proxy.runpod.net"

    # ---- lifecycle ---------------------------------------------------------
    async def refresh(self) -> dict:
        self._meta = await self._rest.get_pod(self.id)
        return self._meta

    async def _wait_provision(self) -> None:
        deadline = time.monotonic() + self.config.provision_timeout.total_seconds()
        while True:
            await self.refresh()
            if self.status in ("EXITED", "TERMINATED"):
                raise ProvisionError(f"pod {self.id} entered terminal state {self.status}")
            if self.status == "RUNNING" and self.host and self.mapped_port(22):
                return
            if time.monotonic() >= deadline:
                raise PodNotReadyError(
                    f"pod {self.id} not RUNNING+routable within "
                    f"{self.config.provision_timeout.total_seconds():.0f}s (status={self.status})"
                )
            await asyncio.sleep(self.config.poll_interval)

    async def _wait_ready(self) -> None:
        await wait_ready(self, self.config.ready, self.config.ready_timeout,
                         self.config.poll_interval)

    async def teardown(self) -> None:
        await self._rest.delete_pod(self.id)

    # ---- exec / transfer: inherited from SshBox ------------------------------
    def _ssh_endpoint(self) -> tuple[str, int]:
        port = self.mapped_port(22)
        if not (self.host and port):
            raise PodNotReadyError("ssh endpoint not available yet")
        return self.host, port


async def _gql_create(config: PodConfig, api_key: str | None) -> dict:
    # GraphQL's gpuTypeId is singular (unlike REST's gpuTypeIds list), so an
    # alias like gpu="A100" is tried candidate-by-candidate, then again on the
    # fallback cloud.
    candidates = config.resolve_gpu_ids()
    clouds = [config.cloud]
    if config.cloud == "COMMUNITY" and config.cloud_fallback:
        clouds.append("SECURE")
    async with RunpodGraphQL(api_key=api_key) as gql:
        # Report every attempt, not just the last: RunPod's GraphQL error
        # strings are opaque ("Something went wrong", "This machine does not
        # have the resources"), and a bare last_err hides that other
        # cloud/GPU combinations were tried and failed differently — which
        # sent issue #27 chasing dockerArgs when the real story was capacity.
        errors: list[str] = []
        for cloud in clouds:
            for gid in candidates:
                gi = config.to_graphql_input(gpu_type_id=gid)
                gi["cloudType"] = cloud
                try:
                    return await gql.create_pod_on_demand(gi)
                except ProvisionError as e:
                    errors.append(f"{gid} on {cloud}: {e}")
        raise ProvisionError(
            "graphql create failed on every cloud/GPU attempt:\n  "
            + "\n  ".join(errors)
        )


async def _create(rest: RunpodRest, config: PodConfig, api_key: str | None) -> str:
    """One create call (GraphQL or REST as the config demands) -> pod id."""
    if config.needs_graphql() and config.resolved_compute == "gpu":
        # Native server-side TTL and the host floors are GraphQL-only
        # (and on-demand = GPU only).
        created = await _gql_create(config, api_key)
    else:
        if config.has_ttl():
            warnings.warn(
                "server-side TTL (stop_after/terminate_after/max_lifetime) is "
                "GPU-only on RunPod; this CPU pod gets NO native timer — if this "
                "process dies, nothing tears the pod down",
                stacklevel=2,
            )
        if config.has_host_floor():
            warnings.warn(
                "min_memory_gb/min_vcpu are GPU-only on RunPod (GraphQL create "
                "path); this CPU pod's host floors are dropped",
                stacklevel=2,
            )
        body = config.to_create_body()
        try:
            created = await rest.create_pod(body)
        except ProvisionError as first:
            if config.cloud == "COMMUNITY" and config.cloud_fallback:
                body["cloudType"] = "SECURE"
                try:
                    created = await rest.create_pod(body)
                except ProvisionError as second:
                    raise ProvisionError(
                        f"create failed on COMMUNITY ({first}) "
                        f"and on the SECURE fallback ({second})"
                    ) from second
            else:
                raise
    pod_id = created.get("id") or created.get("pod", {}).get("id")
    if not pod_id:
        raise ProvisionError(f"could not parse pod id from create response: {created}")
    return pod_id


@contextlib.asynccontextmanager
async def pod(config: PodConfig, *, keep: bool = False,
              api_key: str | None = None) -> AsyncIterator[Pod]:
    """Provision a pod, wait until it's functional, yield it, tear it down.

    On any exception (including a readiness timeout) the pod is still deleted,
    unless ``keep=True`` — or unless the body set ``p.keep = True``, the
    mid-session escape hatch (run() uses it for ``keep_on_failure``).

    With ``config.host_check`` set, a rejected host (the check raising) tears
    that pod down — regardless of ``keep`` — and re-provisions, up to
    ``host_check_retries`` rerolls: RunPod happily re-serves known-bad hosts,
    so "reject and reroll" is the only lever a client has.
    """
    async with RunpodRest(api_key=api_key) as rest:
        rerolls = config.host_check_retries if config.host_check else 0
        rejections: list[str] = []
        for attempt in range(rerolls + 1):
            p = Pod(rest, await _create(rest, config, api_key), config)
            try:
                await p._wait_provision()
                await p._wait_ready()
                if config.host_check:
                    try:
                        await config.host_check(p)
                    except Exception as e:
                        rejections.append(f"pod {p.id} on {p.host}: {e}")
                        # a rejected host is torn down even under keep=True —
                        # nobody wants to keep the pod they just rejected
                        with contextlib.suppress(Exception):
                            await p.teardown()
                        if attempt < rerolls:
                            continue
                        raise ProvisionError(
                            f"host_check rejected every host ({rerolls + 1} attempts):\n  "
                            + "\n  ".join(rejections))
                if config.pip:
                    await install_pip(p, config.pip)
                yield p
                return
            finally:
                if not (keep or p.keep):
                    with contextlib.suppress(Exception):
                        await p.teardown()
