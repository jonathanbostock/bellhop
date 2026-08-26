"""Offline unit tests for bellhop.cluster — mocked GraphQL/REST, no live cluster."""

import asyncio
from datetime import timedelta

import pytest

from bellhop import ClusterConfig, ClusterJobError, PreflightError, ProvisionError
from bellhop.backend import ExecResult
from bellhop.cluster import Cluster, _create_with_bid, _delete_cluster, _discover_ranks


def _cfg(tmp_path, **kw):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test")
    kw.setdefault("gpu", "H100")
    kw.setdefault("ssh_key", str(key))
    return ClusterConfig(**kw)


@pytest.fixture(autouse=True)
def ledger_path(tmp_path, monkeypatch):
    """Every test gets a private membership ledger, never the real XDG one."""
    path = tmp_path / "clusters.jsonl"
    monkeypatch.setenv("BELLHOP_CLUSTER_LEDGER", str(path))
    return path


# ---- config → CreateClusterInput mapping -----------------------------------

def test_input_shape(tmp_path):
    inp = _cfg(tmp_path, nodes=4, gpu_count=8, container_disk_gb=400) \
        .to_graphql_input("NVIDIA H200")
    assert inp["gpuTypeId"] == "NVIDIA H200"
    assert inp["podCount"] == 4
    assert inp["gpuCountPerPod"] == 8
    assert inp["type"] == "TRAINING"
    assert inp["startSsh"] is True
    assert inp["ports"] == "22/tcp"
    assert {"key": "PUBLIC_KEY", "value": "ssh-ed25519 AAAA test"} in inp["env"]
    assert "deployCost" not in inp          # bid is added by the create ladder


def test_gpu_alias_expansion(tmp_path):
    assert _cfg(tmp_path, gpu="A100").resolve_gpu_ids() == [
        "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]


def test_single_node_rejected(tmp_path):
    with pytest.raises(PreflightError, match="nodes >= 2"):
        _cfg(tmp_path, nodes=1)


# ---- create ladder: deployCost auto-bid ------------------------------------

class _FakeGql:
    """Scripted _post: pops the next canned reply (Exception -> raised)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def _post(self, query, variables):
        self.calls.append(variables["input"])
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_autobid_pays_min_times_nodes(tmp_path):
    gql = _FakeGql([
        ProvisionError("graphql error: The requested price 0 is less than the "
                       "current minimum price (3.29) for this type of instance."),
        {"createCluster": {"id": "c1", "pods": [{"id": "p0"}, {"id": "p1"}]}},
    ])
    clu = asyncio.run(_create_with_bid(gql, _cfg(tmp_path, gpu="NVIDIA H100 80GB HBM3")))
    assert clu["id"] == "c1"
    assert gql.calls[1]["deployCost"] == pytest.approx(6.58)   # 3.29 × 2 nodes


def test_autobid_respects_cost_cap(tmp_path):
    gql = _FakeGql([
        ProvisionError("... minimum price (3.29) ..."),
    ])
    with pytest.raises(ProvisionError, match="max_hourly_cost"):
        asyncio.run(_create_with_bid(
            gql, _cfg(tmp_path, gpu="NVIDIA H100 80GB HBM3", max_hourly_cost=5.0)))


def test_stockout_walks_gpu_candidates(tmp_path):
    # A100 alias = two gpuTypeIds; first prices then stocks out, second works
    gql = _FakeGql([
        ProvisionError("... minimum price (1.39) ..."),
        ProvisionError("graphql error: Insufficient resources"),
        ProvisionError("... minimum price (1.59) ..."),
        {"createCluster": {"id": "c2", "pods": [{"id": "a"}, {"id": "b"}]}},
    ])
    clu = asyncio.run(_create_with_bid(gql, _cfg(tmp_path, gpu="A100")))
    assert clu["id"] == "c2"
    assert gql.calls[0]["gpuTypeId"] == "NVIDIA A100 80GB PCIe"
    assert gql.calls[2]["gpuTypeId"] == "NVIDIA A100-SXM4-80GB"
    assert gql.calls[3]["deployCost"] == pytest.approx(3.18)


def test_all_stockout_raises_capacity(tmp_path):
    gql = _FakeGql([
        ProvisionError("... minimum price (3.29) ..."),
        ProvisionError("graphql error: Insufficient resources"),
    ])
    with pytest.raises(ProvisionError, match="no cluster capacity"):
        asyncio.run(_create_with_bid(gql, _cfg(tmp_path, gpu="NVIDIA H100 80GB HBM3")))


# ---- rank discovery / rendezvous env ---------------------------------------

class _FakeNode:
    def __init__(self, pod_id, rank, ip, gpu_count=1, exit_code=0):
        self.id = pod_id
        self._rank, self._ip = rank, ip
        self.exec_env = None
        self._exit = exit_code

        class _C:  # duck-typed .config.gpu_count
            pass
        self.config = _C()
        self.config.gpu_count = gpu_count

    async def exec(self, cmd, env=None, timeout=None):
        self.exec_env = env
        if "proc/1/environ" in cmd:
            return ExecResult(0, f"NODE_RANK={self._rank}\nNODE_ADDR={self._ip}/24\n", "")
        if self._exit:
            return ExecResult(self._exit, "", f"rank {self._rank} boom")
        return ExecResult(0, f"ok rank {self._rank}", "")


def test_discover_ranks_orders_and_strips_cidr():
    pods = [_FakeNode("pB", 1, "10.65.0.3"), _FakeNode("pA", 0, "10.65.0.2")]
    nodes, ips = asyncio.run(_discover_ranks(pods))
    assert [n.id for n in nodes] == ["pA", "pB"]
    assert ips == {0: "10.65.0.2", 1: "10.65.0.3"}


def test_rank_env_derives_rendezvous():
    nodes = [_FakeNode("a", 0, "10.65.0.2", gpu_count=8),
             _FakeNode("b", 1, "10.65.0.3", gpu_count=8)]
    clu = Cluster("cid", nodes, {0: "10.65.0.2", 1: "10.65.0.3"})
    env = clu.rank_env(1)
    assert env["PRIMARY_ADDR"] == env["MASTER_ADDR"] == "10.65.0.2"
    assert env["PRIMARY_PORT"] == env["MASTER_PORT"] == "29500"
    assert env["NODE_RANK"] == "1"
    assert env["NODE_ADDR"] == "10.65.0.3"
    assert env["NUM_NODES"] == "2" and env["NUM_TRAINERS"] == "8"
    assert env["WORLD_SIZE"] == "16"
    assert env["NCCL_SOCKET_IFNAME"] == "ens1"


def test_exec_all_injects_env_and_collects():
    nodes = [_FakeNode("a", 0, "10.65.0.2"), _FakeNode("b", 1, "10.65.0.3")]
    clu = Cluster("cid", nodes, {0: "10.65.0.2", 1: "10.65.0.3"})
    results = asyncio.run(clu.exec_all("echo hi", env={"EXTRA": "1"}))
    assert [r.exit_code for r in results] == [0, 0]
    assert nodes[1].exec_env["NODE_RANK"] == "1"
    assert nodes[1].exec_env["EXTRA"] == "1"


def test_exec_all_failure_raises_cluster_job_error():
    nodes = [_FakeNode("a", 0, "10.65.0.2"), _FakeNode("b", 1, "10.65.0.3", exit_code=3)]
    clu = Cluster("cid", nodes, {0: "10.65.0.2", 1: "10.65.0.3"})
    with pytest.raises(ClusterJobError) as ei:
        asyncio.run(clu.exec_all("boom"))
    assert ei.value.results[1].exit_code == 3
    assert "boom" in ei.value.log_tail or "rank 1" in ei.value.log_tail


# ---- teardown fallback ------------------------------------------------------

class _FakeRest:
    def __init__(self, survivors, undeletable=()):
        self.survivors = set(survivors)
        self.undeletable = set(undeletable)   # deletes silently fail (the orphan bug)
        self.deleted = []

    async def get_pod(self, pid):
        if pid in self.survivors:
            return {"id": pid}
        raise RuntimeError("404")

    async def delete_pod(self, pid):
        self.deleted.append(pid)
        if pid not in self.undeletable:
            self.survivors.discard(pid)

    async def list_pods(self):
        return [{"id": pid} for pid in sorted(self.survivors)]


async def _no_sleep(_):
    return None


def test_delete_cluster_falls_back_to_pod_deletes(monkeypatch):
    async def _run():
        gql = _FakeGql([{"deleteCluster": True}])
        rest = _FakeRest(survivors={"p1"})
        # patch out the cascade-settle sleep so the test is instant
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        remaining = await _delete_cluster(gql, rest, "cid", ["p0", "p1"])
        assert rest.deleted == ["p1"]   # only the survivor
        assert remaining == []          # ...and its death was verified

    asyncio.run(_run())


def test_delete_cluster_names_unkillable_pods_loudly(monkeypatch, capsys):
    # deleteCluster can orphan still-billing member pods; a delete that keeps
    # failing must be shouted about, never suppressed into silence
    async def _run():
        gql = _FakeGql([{"deleteCluster": True}])
        rest = _FakeRest(survivors={"p0", "p1"}, undeletable={"p1"})
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        remaining = await _delete_cluster(gql, rest, "cid", ["p0", "p1"])
        assert remaining == ["p1"]
        assert rest.deleted.count("p1") == 3   # one delete attempt per verify round

    asyncio.run(_run())
    err = capsys.readouterr().err
    assert "RUNNING AND BILLING" in err and "p1" in err


class _GcGql:
    """Constructor stand-in AND instance: gc calls RunpodGraphQL(api_key)."""

    def __init__(self, clusters):
        self.clusters = clusters

    def __call__(self, api_key=None):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def _post(self, q, v):
        return {"myself": {"clusters": self.clusters}}


class _GcRest:
    def __init__(self, alive=(), undeletable=(), linkage=None):
        self.alive = set(alive)
        self.undeletable = set(undeletable)
        self.linkage = linkage or {}        # pod id -> REST clusterId field
        self.deleted = []

    def __call__(self, api_key=None):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def get_pod(self, pid):
        if pid in self.alive:
            return {"id": pid, "name": f"n-{pid}"}
        raise RuntimeError("404")

    async def delete_pod(self, pid):
        self.deleted.append(pid)
        if pid not in self.undeletable:
            self.alive.discard(pid)

    async def list_pods(self):
        return [{"id": pid, "name": f"n-{pid}",
                 **({"clusterId": self.linkage[pid]} if pid in self.linkage else {})}
                for pid in sorted(self.alive)]


def _patch_gc(monkeypatch, gql, rest):
    import importlib

    clumod = importlib.import_module("bellhop.cluster")
    monkeypatch.setattr(clumod, "RunpodGraphQL", gql)
    monkeypatch.setattr(clumod, "RunpodRest", rest)
    return clumod


_LIVE_C = {"id": "live-c", "name": "live", "createdAt": "2026-08-26T00:00:00Z",
           "gpuTypeId": "NVIDIA H200", "podCount": 2, "pods": []}
_FAR = timedelta(hours=10_000)   # threshold so live clusters are never age-reaped


def test_ledger_roundtrip(ledger_path):
    from bellhop.cluster import _ledger_forget, _ledger_load, _ledger_record

    _ledger_record("c1", ["a", "b"], "one")
    _ledger_record("c2", ["c"], "two")
    _ledger_record("c1", ["b"], "one")           # narrowing append: last write wins
    entries = {e["cluster_id"]: e for e in _ledger_load()}
    assert entries["c1"]["pod_ids"] == ["b"]
    assert entries["c2"]["pod_ids"] == ["c"]
    _ledger_forget("c1")
    assert [e["cluster_id"] for e in _ledger_load()] == ["c2"]
    with open(ledger_path, "a") as f:            # torn/foreign lines are tolerated
        f.write("{not json\n")
    assert [e["cluster_id"] for e in _ledger_load()] == ["c2"]


def test_gc_ledger_sweep_reaps_then_prunes(monkeypatch):
    # The exact orphan record is OUR ledger (REST pods carry no cluster field):
    # a dead cluster's still-alive pods are reaped at any age; the entry is
    # dropped only once a later round verifies every pod gone.
    rest = _GcRest(alive={"o1"})
    clumod = _patch_gc(monkeypatch, _GcGql([_LIVE_C]), rest)
    clumod._ledger_record("dead-c", ["o1", "o2"], "gone")

    reaped = asyncio.run(clumod.gc_clusters(_FAR))
    assert [r["id"] for r in reaped] == ["o1"]
    assert reaped[0]["orphaned_pod_of"] == "dead-c"
    assert rest.deleted == ["o1"]
    # this round's delete was unverified, so the entry survives...
    assert [e["cluster_id"] for e in clumod._ledger_load()] == ["dead-c"]
    # ...and the next round finds every pod dead and prunes it
    assert asyncio.run(clumod.gc_clusters(_FAR)) == []
    assert clumod._ledger_load() == []


def test_gc_ledger_sweep_dry_run_touches_nothing(monkeypatch):
    rest = _GcRest(alive={"o1"})
    clumod = _patch_gc(monkeypatch, _GcGql([_LIVE_C]), rest)
    clumod._ledger_record("dead-c", ["o1"], "gone")

    reaped = asyncio.run(clumod.gc_clusters(_FAR, dry_run=True))
    assert [r["id"] for r in reaped] == ["o1"] and rest.deleted == []
    assert [e["cluster_id"] for e in clumod._ledger_load()] == ["dead-c"]


def test_gc_ledger_skips_live_clusters(monkeypatch):
    rest = _GcRest(alive={"m1"})
    clumod = _patch_gc(monkeypatch, _GcGql([_LIVE_C]), rest)
    clumod._ledger_record("live-c", ["m1"], "live")

    assert asyncio.run(clumod.gc_clusters(_FAR)) == []
    assert rest.deleted == []
    assert [e["cluster_id"] for e in clumod._ledger_load()] == ["live-c"]


def test_gc_adopts_survivors_of_reaped_cluster(monkeypatch):
    # An age-reaped cluster whose deleteCluster orphans a pod: the survivor is
    # written to the ledger (even though this machine never created it), so
    # the next gc round can finish the job after the cluster object is gone.
    old = {"id": "old-c", "name": "old", "createdAt": "2026-08-01T00:00:00Z",
           "gpuTypeId": "NVIDIA H200", "podCount": 1, "pods": [{"id": "s1"}]}
    rest = _GcRest(alive={"s1"}, undeletable={"s1"})
    clumod = _patch_gc(monkeypatch, _GcGql([old]), rest)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    reaped = asyncio.run(clumod.gc_clusters(timedelta(0)))
    assert [r["id"] for r in reaped] == ["old-c"]
    entries = clumod._ledger_load()
    assert entries[0]["cluster_id"] == "old-c" and entries[0]["pod_ids"] == ["s1"]


def test_gc_linkage_fallback_still_reaps_when_rest_reports_it(monkeypatch):
    # Belt-and-braces: live probes show REGULAR pods have no cluster field,
    # but member pods might. If REST ever does report linkage, orphans are
    # reaped even with an empty ledger (e.g. created from another machine).
    rest = _GcRest(alive={"member", "orphan", "plain"},
                   linkage={"member": "live-c", "orphan": "dead-c"})
    clumod = _patch_gc(monkeypatch, _GcGql([_LIVE_C]), rest)

    reaped = asyncio.run(clumod.gc_clusters(_FAR))
    assert [r["id"] for r in reaped] == ["orphan"] and rest.deleted == ["orphan"]
    assert reaped[0]["orphaned_pod_of"] == "dead-c"

    rest.deleted.clear()
    rest.alive.add("orphan")
    reaped = asyncio.run(clumod.gc_clusters(_FAR, dry_run=True))
    assert [r["id"] for r in reaped] == ["orphan"] and rest.deleted == []


# ---- ledger lifecycle through the cluster() context manager -----------------

def _patch_cm(monkeypatch, clumod, create_reply, delete_result):
    """Stub everything around cluster() so only its ledger handling is real."""

    class _Noop:
        def __init__(self, api_key=None):
            pass

        async def aclose(self):
            pass

    class _P:
        def __init__(self, rest, pid, cfg):
            self.id, self.config = pid, cfg

        async def _wait_provision(self):
            pass

        async def _wait_ready(self):
            pass

    async def _create(gql, config):
        return create_reply

    deletes = []

    async def _delete(gql, rest, cid, pids):
        deletes.append((cid, list(pids)))
        return list(delete_result)

    async def _discover(pods):
        return pods, {i: f"10.65.0.{i + 2}" for i in range(len(pods))}

    monkeypatch.setattr(clumod, "RunpodGraphQL", _Noop)
    monkeypatch.setattr(clumod, "RunpodRest", _Noop)
    monkeypatch.setattr(clumod, "Pod", _P)
    monkeypatch.setattr(clumod, "_create_with_bid", _create)
    monkeypatch.setattr(clumod, "_delete_cluster", _delete)
    monkeypatch.setattr(clumod, "_discover_ranks", _discover)
    return deletes


def test_cluster_cm_records_at_birth_and_forgets_on_clean_teardown(tmp_path, monkeypatch):
    import importlib

    clumod = importlib.import_module("bellhop.cluster")
    deletes = _patch_cm(monkeypatch, clumod,
                        {"id": "c9", "pods": [{"id": "x"}, {"id": "y"}]}, [])
    inside = {}

    async def _run():
        async with clumod.cluster(_cfg(tmp_path)):
            inside["entries"] = clumod._ledger_load()

    asyncio.run(_run())
    assert inside["entries"][0]["cluster_id"] == "c9"
    assert inside["entries"][0]["pod_ids"] == ["x", "y"]
    assert deletes == [("c9", ["x", "y"])]
    assert clumod._ledger_load() == []       # zero survivors → entry dropped


def test_cluster_cm_narrows_ledger_to_teardown_survivors(tmp_path, monkeypatch):
    import importlib

    clumod = importlib.import_module("bellhop.cluster")
    _patch_cm(monkeypatch, clumod,
              {"id": "c9", "pods": [{"id": "x"}, {"id": "y"}]}, ["y"])

    async def _run():
        async with clumod.cluster(_cfg(tmp_path)):
            pass

    asyncio.run(_run())
    entries = clumod._ledger_load()
    assert entries[0]["cluster_id"] == "c9" and entries[0]["pod_ids"] == ["y"]
