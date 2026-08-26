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
    def __init__(self, survivors):
        self.survivors = set(survivors)
        self.deleted = []

    async def get_pod(self, pid):
        if pid in self.survivors:
            return {"id": pid}
        raise RuntimeError("404")

    async def delete_pod(self, pid):
        self.deleted.append(pid)


def test_delete_cluster_falls_back_to_pod_deletes(monkeypatch):
    async def _run():
        gql = _FakeGql([{"deleteCluster": True}])
        rest = _FakeRest(survivors={"p1"})
        # patch out the cascade-settle sleep so the test is instant
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        await _delete_cluster(gql, rest, "cid", ["p0", "p1"])
        assert rest.deleted == ["p1"]   # only the survivor

    async def _no_sleep(_):
        return None

    asyncio.run(_run())
