"""RunPod Instant Clusters — N coordinated nodes for multi-node training.

A :class:`Cluster` is N :class:`~bellhop.pod.Pod` channels (one per node,
indexed by ``NODE_RANK``) plus cluster identity, so everything proven for
single pods (ssh exec, tar push/pull, readiness probes) carries over
unchanged. Cluster CRUD is GraphQL-only — the REST v1 API has no cluster
endpoints (see docs/design/instant-clusters.md for the recovered contract and
the M0 live-probe findings this module encodes):

- ``deployCost`` prices the WHOLE cluster; the server divides by ``podCount``
  and compares to a per-node minimum it only reveals in the rejection error.
  We bid, parse the minimum, and re-bid ``min × podCount`` (capped by
  ``max_hourly_cost``).
- RunPod injects ``NODE_RANK`` / ``NODE_ADDR`` / ``NUM_NODES`` /
  ``NUM_TRAINERS`` / ``WORLD_SIZE`` into PID-1's env only (not ssh sessions),
  and — despite its docs — does NOT inject ``PRIMARY_ADDR``/``MASTER_*``.
  :meth:`Cluster.exec_all` therefore injects the full documented set itself,
  deriving the rendezvous from rank-0's ``NODE_ADDR`` + a fixed port.
- There is no server-side TTL for clusters. Teardown is client-owned:
  the context manager always deletes (with per-pod fallback), a watchdog
  enforces ``max_lifetime``, and ``bellhop clusters gc`` reaps leaks.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from .backend import ExecResult
from .errors import (
    PodNotReadyError,
    PreflightError,
    ProvisionError,
    RemoteJobError,
    ResultsMissingError,
    is_capacity_error,
)
from .graphql import RunpodGraphQL
from .pod import DEFAULT_GPU_IMAGE, Pod, PodConfig
from .probes import ReadyProbe, SshProbe
from .rest import RunpodRest
from .run import RunSpec, RunResult, _gcs_upload, _is_git, _job_script, _tail

# vars RunPod actually injects into PID-1 (M0-verified; PRIMARY_*/MASTER_* are
# documented but absent in practice)
_INJECTED = ("NODE_RANK", "NODE_ADDR", "NUM_NODES", "NUM_TRAINERS", "WORLD_SIZE")

DEFAULT_RDZV_PORT = 29500

_CREATE_CLUSTER = """
mutation createCluster($input: CreateClusterInput!) {
  createCluster(input: $input) { id name type gpuTypeId podCount gpuCountPerPod pods { id } }
}
"""
_DELETE_CLUSTER = """
mutation deleteCluster($input: DeleteClusterInput!) { deleteCluster(input: $input) }
"""
_LIST_CLUSTERS = """
query { myself { clusters { id name type gpuTypeId podCount gpuCountPerPod createdAt pods { id } } } }
"""

_MIN_PRICE_RE = re.compile(r"minimum price \(([\d.]+)\)")


@dataclass
class ClusterConfig:
    """Shape of an Instant Cluster; the multi-node sibling of PodConfig."""

    gpu: str                              # canonical short name or full gpuTypeId
    nodes: int = 2                        # → podCount (RunPod sells 2-8)
    gpu_count: int = 8                    # → gpuCountPerPod
    image: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    container_disk_gb: int = 100
    volume_gb: int = 0
    volume_mount_path: str = "/workspace"
    network_volume_id: str | None = None
    data_center_id: str | None = None
    allowed_cuda_versions: list[str] | None = None
    # $/hr cap for the WHOLE cluster; the auto-bid pays the leaked per-node
    # minimum × nodes but never exceeds this. None = pay whatever the current
    # minimum is (the minimums track on-demand pod pricing).
    max_hourly_cost: float | None = None
    rendezvous_port: int = DEFAULT_RDZV_PORT
    # auth / connection (per-node, same as PodConfig)
    ssh_key: str | None = None
    ssh_user: str = "root"
    ready: ReadyProbe = field(default_factory=lambda: SshProbe("true"))
    provision_timeout: timedelta = timedelta(seconds=900)
    ready_timeout: timedelta = timedelta(seconds=900)
    poll_interval: float = 8.0
    # client-side hard cap — there is NO server-side TTL for clusters
    max_lifetime: timedelta = timedelta(hours=24)
    name: str = "bellhop"                 # local bookkeeping only (no API field)

    def __post_init__(self):
        if self.nodes < 2:
            raise PreflightError("a cluster needs nodes >= 2 (use PodConfig for one box)")

    def _node_pod_config(self) -> PodConfig:
        """Per-node PodConfig carrying the ssh/probe/timeout settings."""
        return PodConfig(
            gpu=self.gpu, gpu_count=self.gpu_count,   # gpu= gets alias expansion
            ssh_key=self.ssh_key, ssh_user=self.ssh_user, ready=self.ready,
            provision_timeout=self.provision_timeout,
            ready_timeout=self.ready_timeout, poll_interval=self.poll_interval,
            name=self.name,
        )

    def resolve_gpu_ids(self) -> list[str]:
        return self._node_pod_config().resolve_gpu_ids()

    def to_graphql_input(self, gpu_type_id: str) -> dict[str, Any]:
        inp: dict[str, Any] = {
            "gpuTypeId": gpu_type_id,
            "podCount": self.nodes,
            "gpuCountPerPod": self.gpu_count,
            "type": "TRAINING",
            "imageName": self.image or DEFAULT_GPU_IMAGE,
            "containerDiskInGb": self.container_disk_gb,
            "ports": "22/tcp",
            "startSsh": True,
            "env": [{"key": k, "value": v} for k, v in self._create_env().items()],
        }
        if self.volume_gb:
            inp["volumeInGb"] = self.volume_gb
            inp["volumeMountPath"] = self.volume_mount_path
        if self.network_volume_id:
            inp["networkVolumeId"] = self.network_volume_id
        if self.data_center_id:
            inp["dataCenterId"] = self.data_center_id
        if self.allowed_cuda_versions:
            inp["allowedCudaVersions"] = self.allowed_cuda_versions
        return inp

    def _create_env(self) -> dict[str, str]:
        env = dict(self.env)
        env.setdefault("PUBLIC_KEY", self._node_pod_config().pubkey_text())
        return env


class ClusterJobError(RemoteJobError):
    """A cluster-wide exec failed on at least one rank."""

    def __init__(self, message: str, *, results: dict[int, ExecResult | None]):
        failed = {r: res.exit_code for r, res in results.items()
                  if res is not None and res.exit_code != 0}
        first_bad = min(failed) if failed else -1
        tail = ""
        if first_bad >= 0 and results[first_bad] is not None:
            res = results[first_bad]
            tail = (res.stderr or res.stdout)[-2000:]
        super().__init__(f"{message} (failed ranks: {sorted(failed) or 'none — cancelled'})",
                         remote_exit=failed.get(first_bad, -1), log_tail=tail)
        self.results = results


class Cluster:
    """A live Instant Cluster. Construct via :func:`cluster`."""

    def __init__(self, cluster_id: str, nodes: list[Pod], node_ips: dict[int, str],
                 rendezvous_port: int = DEFAULT_RDZV_PORT):
        self.id = cluster_id
        self.nodes = nodes                     # index == NODE_RANK
        self.node_ips = node_ips               # rank -> overlay IP (no CIDR suffix)
        self.rendezvous_port = rendezvous_port

    @property
    def primary(self) -> Pod:
        return self.nodes[0]

    def rank_env(self, rank: int) -> dict[str, str]:
        """The full documented cluster env for one rank.

        Self-derived rather than read from the pod: RunPod puts its subset in
        PID-1's env only (invisible to ssh sessions) and omits PRIMARY_*
        entirely, so injecting our own copy is both necessary and sufficient.
        """
        primary = self.node_ips[0]
        n = len(self.nodes)
        per = str(self.nodes[rank].config.gpu_count)
        return {
            "PRIMARY_ADDR": primary, "MASTER_ADDR": primary,
            "PRIMARY_PORT": str(self.rendezvous_port), "MASTER_PORT": str(self.rendezvous_port),
            "HOST_NODE_ADDR": f"{primary}:{self.rendezvous_port}",
            "NODE_ADDR": self.node_ips[rank],
            "NODE_RANK": str(rank),
            "NUM_NODES": str(n),
            "NUM_TRAINERS": per,
            "WORLD_SIZE": str(n * int(per)),
            # inter-node traffic must use the overlay NICs, never eth0
            "NCCL_SOCKET_IFNAME": "ens1",
        }

    async def exec_all(self, cmd: str, *, env: dict[str, str] | None = None,
                       timeout: float | None = None) -> list[ExecResult]:
        """Run ``cmd`` on every rank concurrently, with the cluster env injected.

        First non-zero exit cancels the sibling execs (with a static rendezvous
        the survivors would hang at their next collective) and raises
        :class:`ClusterJobError` carrying the per-rank results.
        """
        async def _one(rank: int) -> ExecResult:
            merged = {**self.rank_env(rank), **(env or {})}
            res = await self.nodes[rank].exec(cmd, env=merged, timeout=timeout)
            if res.exit_code != 0:
                raise _RankFailed(rank, res)
            return res

        tasks = [asyncio.create_task(_one(r)) for r in range(len(self.nodes))]
        results: dict[int, ExecResult | None] = {r: None for r in range(len(self.nodes))}
        try:
            for coro in asyncio.as_completed(tasks):
                await coro
        except _RankFailed as f:
            results[f.rank] = f.result
            for t in tasks:
                t.cancel()
            done = await asyncio.gather(*tasks, return_exceptions=True)
            for r, item in enumerate(done):
                if isinstance(item, ExecResult):
                    results[r] = item
                elif isinstance(item, _RankFailed):
                    results[item.rank] = item.result
            raise ClusterJobError(f"exec_all failed on cluster {self.id}", results=results) from None
        for r, t in enumerate(tasks):
            results[r] = t.result()
        return [results[r] for r in range(len(self.nodes))]  # type: ignore[misc]

    async def push_all(self, local: str | Path, remote: str) -> None:
        await asyncio.gather(*(p.push(local, remote) for p in self.nodes))

    async def pull(self, remote: str, local_dest: str | Path, *, rank: int = 0) -> None:
        await self.nodes[rank].pull(remote, local_dest)


class _RankFailed(Exception):
    def __init__(self, rank: int, result: ExecResult):
        self.rank, self.result = rank, result


async def _create_with_bid(gql: RunpodGraphQL, config: ClusterConfig) -> dict[str, Any]:
    """Walk the GPU-candidate ladder; auto-bid the leaked per-node minimum."""
    last: Exception | None = None
    for gpu_id in config.resolve_gpu_ids():
        inp = config.to_graphql_input(gpu_id)
        for _ in range(2):
            try:
                data = await gql._post(_CREATE_CLUSTER, {"input": inp})
                return data["createCluster"]
            except Exception as e:  # noqa: BLE001 — walk past stock/price rejections
                last = e
                m = _MIN_PRICE_RE.search(str(e))
                if m and "deployCost" not in inp:
                    per_node = float(m.group(1))
                    total = round(per_node * config.nodes, 2)
                    if config.max_hourly_cost is not None and total > config.max_hourly_cost:
                        raise ProvisionError(
                            f"cluster minimum is ${per_node}/node-hr → ${total}/hr total, "
                            f"over max_hourly_cost=${config.max_hourly_cost}") from e
                    inp["deployCost"] = total
                    continue
                break  # stock-out or other error → next GPU candidate
    if last is not None and is_capacity_error(last):
        raise ProvisionError(f"no cluster capacity for {config.gpu}: {last}") from last
    raise ProvisionError(f"createCluster failed: {last}") from last


async def _discover_ranks(pods: list[Pod]) -> tuple[list[Pod], dict[int, str]]:
    """Read NODE_RANK/NODE_ADDR from each pod's PID-1 env; order by rank."""
    probe = 'tr "\\0" "\\n" < /proc/1/environ | grep -E "^(NODE_RANK|NODE_ADDR)="'
    outs = await asyncio.gather(*(p.exec(probe, timeout=60) for p in pods))
    by_rank: dict[int, Pod] = {}
    ips: dict[int, str] = {}
    for p, res in zip(pods, outs):
        rank_m = re.search(r"NODE_RANK=(\d+)", res.stdout)
        addr_m = re.search(r"NODE_ADDR=([\d.]+)", res.stdout)  # strips /24 suffix
        if not (rank_m and addr_m):
            raise PodNotReadyError(
                f"pod {p.id} did not expose NODE_RANK/NODE_ADDR in /proc/1/environ: "
                f"{res.stdout[:200]!r}")
        rank = int(rank_m.group(1))
        by_rank[rank] = p
        ips[rank] = addr_m.group(1)
    if sorted(by_rank) != list(range(len(pods))):
        raise PodNotReadyError(f"cluster ranks incomplete: {sorted(by_rank)}")
    return [by_rank[r] for r in range(len(pods))], ips


async def _lifetime_watchdog(clu: Cluster, gql: RunpodGraphQL, rest: RunpodRest,
                             lifetime: timedelta) -> None:
    # tears the cluster down out from under any still-running exec — its ssh
    # sessions die and exec_all fails, which is the intended failure mode
    await asyncio.sleep(lifetime.total_seconds())
    import sys
    print(f"bellhop: cluster {clu.id} hit max_lifetime {lifetime} — tearing down",
          file=sys.stderr, flush=True)
    # Grace hook: salvage rank 0's results before destruction (run_cluster
    # sets it) — a run once died to its own 3h timer DURING the results pull.
    grace = getattr(clu, "on_lifetime_expiry", None)
    if grace is not None:
        with contextlib.suppress(Exception):
            from .sshbox import LIFETIME_GRACE_SECONDS
            await asyncio.wait_for(grace(), LIFETIME_GRACE_SECONDS)
    with contextlib.suppress(Exception):
        await _delete_cluster(gql, rest, clu.id, [p.id for p in clu.nodes])


async def _delete_cluster(gql: RunpodGraphQL, rest: RunpodRest,
                          cluster_id: str, pod_ids: list[str]) -> None:
    """deleteCluster, then verify the cascade; fall back to per-pod deletes."""
    with contextlib.suppress(Exception):
        await gql._post(_DELETE_CLUSTER, {"input": {"id": cluster_id}})
    await asyncio.sleep(5)
    for pid in pod_ids:
        try:
            await rest.get_pod(pid)
        except Exception:
            continue          # gone — the normal case (M0: cascade works)
        with contextlib.suppress(Exception):
            await rest.delete_pod(pid)


@contextlib.asynccontextmanager
async def cluster(config: ClusterConfig, *, api_key: str | None = None):
    """Provision an Instant Cluster, yield a :class:`Cluster`, always tear down."""
    config._node_pod_config().resolve_ssh_key()   # preflight before spending money
    gql = RunpodGraphQL(api_key)
    rest = RunpodRest(api_key)
    try:
        created = await _create_with_bid(gql, config)
        pod_ids = [p["id"] for p in created["pods"]]
        try:
            node_cfg = config._node_pod_config()
            pods = [Pod(rest, pid, node_cfg) for pid in pod_ids]
            await asyncio.gather(*(p._wait_provision() for p in pods))
            await asyncio.gather(*(p._wait_ready() for p in pods))
            nodes, ips = await _discover_ranks(pods)
            clu = Cluster(created["id"], nodes, ips, config.rendezvous_port)
            watchdog = asyncio.create_task(
                _lifetime_watchdog(clu, gql, rest, config.max_lifetime))
            try:
                yield clu
            finally:
                watchdog.cancel()
        finally:
            await _delete_cluster(gql, rest, created["id"], pod_ids)
    finally:
        await gql.aclose()
        await rest.aclose()


async def run_cluster(spec: RunSpec, config: ClusterConfig, *,
                      api_key: str | None = None) -> RunResult:
    """One-shot multi-node pipeline; the N-node sibling of :func:`bellhop.run`.

    Pushes the codebase to every node, runs setup and the job on every rank
    concurrently (cluster env injected — the job command is expected to be a
    ``torchrun --node_rank $NODE_RANK ... --rdzv_endpoint
    $PRIMARY_ADDR:$PRIMARY_PORT`` invocation or equivalent), pulls
    ``results_subdir`` from rank 0, optionally uploads to GCS.
    """
    if not (spec.slug and spec.codebase and spec.run):
        raise PreflightError("slug, codebase and run are all required")
    if _is_git(spec.codebase):
        raise PreflightError("run_cluster needs a local codebase dir (git URL not yet supported)")
    if not Path(spec.codebase).is_dir():
        raise PreflightError(f"codebase dir not found: {spec.codebase}")

    local_out = spec.local_out or os.path.join(os.getcwd(), "experiments", spec.slug)
    Path(local_out).mkdir(parents=True, exist_ok=True)
    run_dir = f"/workspace/{spec.slug}"
    results_remote = f"{run_dir}/{spec.results_subdir}"

    async with cluster(config, api_key=api_key) as clu:
        # Salvage rank 0's results if the max_lifetime watchdog fires mid-run.
        async def _salvage():
            if await clu.primary.exists_remote(results_remote):
                await clu.pull(results_remote, local_out)
        clu.on_lifetime_expiry = _salvage

        await clu.exec_all(f"mkdir -p {shlex.quote(run_dir)}")
        await clu.push_all(spec.codebase, run_dir)
        job_results = await clu.exec_all(_job_script(spec, run_dir),
                                         env=spec.env, timeout=spec.timeout)
        if await clu.primary.exists_remote(results_remote):
            await clu.pull(results_remote, local_out)
        else:
            raise ResultsMissingError(f"job succeeded but no results dir at {results_remote}")

        gcs_uri = retrieve_cmd = None
        if spec.gcs_base:
            gcs_uri = spec.gcs_base.rstrip("/") + f"/{spec.slug}/"
            await _gcs_upload(local_out, gcs_uri)
            retrieve_cmd = f"gcloud storage cp -r {gcs_uri} ./"

        pulled_dir = os.path.basename(results_remote.rstrip("/"))
        return RunResult(
            slug=spec.slug, pod_id=clu.id,
            remote_exit=max(r.exit_code for r in job_results),
            local_results=local_out, gcs_uri=gcs_uri, retrieve_cmd=retrieve_cmd,
            log_tail=_tail(os.path.join(local_out, pulled_dir, "run.log")),
        )


async def list_clusters(api_key: str | None = None) -> list[dict[str, Any]]:
    async with RunpodGraphQL(api_key) as gql:
        data = await gql._post(_LIST_CLUSTERS, {})
        return data["myself"]["clusters"]


async def gc_clusters(older_than: timedelta, *, api_key: str | None = None,
                      dry_run: bool = False) -> list[dict[str, Any]]:
    """Reap clusters older than ``older_than`` (the missing server-side TTL)."""
    from datetime import datetime, timezone

    def _parse_created(raw: str) -> datetime:
        # RunPod timestamps come as ISO-with-Z or "YYYY-MM-DD HH:MM:SS"; both UTC
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00").replace(" ", "T"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    reaped = []
    async with RunpodGraphQL(api_key) as gql, RunpodRest(api_key) as rest:
        data = await gql._post(_LIST_CLUSTERS, {})
        for clu in data["myself"]["clusters"]:
            age = datetime.now(timezone.utc) - _parse_created(clu["createdAt"])
            if age >= older_than:
                if not dry_run:
                    await _delete_cluster(gql, rest, clu["id"],
                                          [p["id"] for p in clu.get("pods") or []])
                reaped.append({**clu, "age_hours": round(age.total_seconds() / 3600, 2)})
    return reaped
