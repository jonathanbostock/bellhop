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
import json
import os
import re
import shlex
import sys
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


# ---- membership ledger -------------------------------------------------------
# RunPod's REST pod records carry NO cluster linkage (live-probed 2026-08-26:
# a pod's complete key set has nothing cluster-ish in it), so once deleteCluster
# kills the cluster OBJECT its orphaned member pods are unattributable via the
# API. The only exact record of membership is createCluster's own response —
# persist it locally at birth, and let gc reap from it. Lives in XDG *state*,
# not cache: wiping a cache must never lose the only record of billing pods.

_LEDGER_ENV = "BELLHOP_CLUSTER_LEDGER"


def _ledger_path() -> Path:
    if os.environ.get(_LEDGER_ENV):
        return Path(os.environ[_LEDGER_ENV]).expanduser()
    state = Path(os.environ.get("XDG_STATE_HOME") or "~/.local/state").expanduser()
    return state / "bellhop" / "clusters.jsonl"


def _ledger_record(cluster_id: str, pod_ids: list[str], name: str) -> None:
    """Append a membership record (last write per cluster wins on load).

    Failure is non-fatal — the cluster is already billing and must proceed —
    but LOUD, because it means gc cannot see these pods if teardown orphans
    them.
    """
    from datetime import datetime, timezone
    try:
        path = _ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"cluster_id": cluster_id, "pod_ids": pod_ids,
                                "name": name,
                                "created": datetime.now(timezone.utc).isoformat()}) + "\n")
    except OSError as e:
        print(f"bellhop: could not record cluster {cluster_id} in {_ledger_path()} "
              f"({e}) — `bellhop clusters gc` will NOT see its pods if teardown "
              f"orphans them; note these ids: {', '.join(pod_ids)}",
              file=sys.stderr, flush=True)


def _ledger_load() -> list[dict[str, Any]]:
    try:
        raw = _ledger_path().read_text()
    except OSError:
        return []
    entries: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        with contextlib.suppress(Exception):    # tolerate torn/foreign lines
            entry = json.loads(line)
            entries[entry["cluster_id"]] = entry
    return list(entries.values())


def _ledger_forget(cluster_id: str) -> None:
    """Drop one cluster's record — call only once every pod is verified gone.

    Best-effort compaction: a concurrent writer can race the rewrite, and the
    worst case is a stale entry whose pods the next gc re-verifies as gone.
    """
    with contextlib.suppress(OSError):
        path = _ledger_path()
        if not path.exists():
            return
        entries = [e for e in _ledger_load() if e["cluster_id"] != cluster_id]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in entries))
        tmp.replace(path)


async def _delete_cluster(gql: RunpodGraphQL, rest: RunpodRest,
                          cluster_id: str, pod_ids: list[str]) -> list[str]:
    """deleteCluster, then verify the cascade; fall back to per-pod deletes.

    The mutation is treated as ADVISORY: it can delete the cluster *object*
    and orphan still-billing member pods (field report: a "deleted" 2×4×H200
    cluster left two $22/hr pods running for 3.5h — and the runaway spend
    saturated the account's spend cap, which then masqueraded as a capacity
    drought). So every member pod is verified gone, deleted directly when
    not, re-verified over three rounds, and any survivor is named LOUDLY —
    a silent suppress here is an open-ended bill. Returns surviving pod ids
    (empty = everything confirmed dead).
    """
    with contextlib.suppress(Exception):
        await gql._post(_DELETE_CLUSTER, {"input": {"id": cluster_id}})
    survivors = list(pod_ids)
    for wait in (5, 15, 30):
        await asyncio.sleep(wait)
        alive = []
        for pid in survivors:
            try:
                await rest.get_pod(pid)
            except Exception:
                continue          # gone — the normal case (M0: cascade works)
            alive.append(pid)
            with contextlib.suppress(Exception):
                await rest.delete_pod(pid)
        if not alive:
            return []
        survivors = alive
    # the last round's deletes were not re-verified — check before shouting
    remaining = []
    for pid in survivors:
        try:
            await rest.get_pod(pid)
            remaining.append(pid)
        except Exception:
            continue
    if remaining:
        print(f"bellhop: cluster {cluster_id} teardown left pods RUNNING AND "
              f"BILLING: {', '.join(remaining)} — the cluster object may already "
              "be gone (so `bellhop clusters gc` orphan sweep or the RunPod "
              "console are the levers); delete them NOW",
              file=sys.stderr, flush=True)
    return remaining


@contextlib.asynccontextmanager
async def cluster(config: ClusterConfig, *, api_key: str | None = None):
    """Provision an Instant Cluster, yield a :class:`Cluster`, always tear down."""
    config._node_pod_config().resolve_ssh_key()   # preflight before spending money
    gql = RunpodGraphQL(api_key)
    rest = RunpodRest(api_key)
    try:
        created = await _create_with_bid(gql, config)
        pod_ids = [p["id"] for p in created["pods"]]
        # ledger BEFORE any await that can fail: a crash between create and
        # here would leave pods only the RunPod console can attribute
        _ledger_record(created["id"], pod_ids, config.name)
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
            survivors = await _delete_cluster(gql, rest, created["id"], pod_ids)
            if survivors:   # narrow the entry so gc re-checks only what's left
                _ledger_record(created["id"], survivors, config.name)
            else:
                _ledger_forget(created["id"])
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
    """Reap clusters older than ``older_than`` (the missing server-side TTL),
    plus — at any age — member pods orphaned by a dead cluster object, using
    the local membership ledger (see ``_ledger_record``)."""
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
                    survivors = await _delete_cluster(
                        gql, rest, clu["id"], [p["id"] for p in clu.get("pods") or []])
                    if survivors:   # adopt into the ledger even if created elsewhere
                        _ledger_record(clu["id"], survivors, clu.get("name", "?"))
                    else:
                        _ledger_forget(clu["id"])
                reaped.append({**clu, "age_hours": round(age.total_seconds() / 3600, 2)})
        # Orphan sweep: deleteCluster can kill the cluster OBJECT and leave its
        # member pods running (see _delete_cluster) — and once the object is
        # gone, RunPod cannot say which pods were members (live-probed: REST
        # pod records carry no cluster field). The exact record is our own
        # create-time ledger: every entry whose cluster no longer exists is
        # checked pod-by-pod, ignoring older_than — its rendezvous peer group
        # is gone, so it is garbage at any age. An entry is dropped only once
        # a round finds every listed pod already dead.
        live = {clu["id"] for clu in data["myself"]["clusters"]}
        swept: set[str] = set()
        for entry in _ledger_load():
            if entry["cluster_id"] in live:
                continue
            found_alive = False
            for pid in entry["pod_ids"]:
                try:
                    pd = await rest.get_pod(pid)
                except Exception:
                    continue          # already dead — the normal case
                found_alive = True
                swept.add(pid)
                if not dry_run:
                    with contextlib.suppress(Exception):
                        await rest.delete_pod(pid)
                reaped.append({"id": pid, "orphaned_pod_of": entry["cluster_id"],
                               "name": pd.get("name", "?")})
            if not dry_run and not found_alive:
                _ledger_forget(entry["cluster_id"])
        # Belt-and-braces: the regular-pod probe can't rule out cluster MEMBER
        # pods carrying a linkage field regular pods omit. If they do, this
        # also catches orphans other machines' ledgers know about; if not, it
        # quietly finds nothing.
        for pd in await rest.list_pods():
            linked = pd.get("clusterId") or pd.get("instantClusterId")
            if not linked or linked in live or pd["id"] in swept:
                continue
            if not dry_run:
                with contextlib.suppress(Exception):
                    await rest.delete_pod(pd["id"])
            reaped.append({"id": pd["id"], "orphaned_pod_of": linked,
                           "name": pd.get("name", "?")})
    return reaped
