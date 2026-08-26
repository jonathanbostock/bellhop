"""Offline unit tests for the Lambda Cloud backend — no live instance, no cost.

The REST client is exercised against httpx.MockTransport (swapped in under
LambdaRest's normal constructor), with the process-wide rate pacer zeroed so
the suite doesn't sleep out the real 12s launch spacing.
"""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest

from bellhop import LambdaConfig, PreflightError, ProvisionError, is_capacity_error
from bellhop.lambda_box import (
    LambdaInstance,
    LambdaRest,
    _ensure_ssh_key,
    _launch,
    _pubkey_blob,
    gc_instances,
)
from bellhop.sshbox import stamp_epoch as _stamp_epoch


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr(LambdaRest, "min_request_interval", 0.0)
    monkeypatch.setattr(LambdaRest, "min_launch_interval", 0.0)


@pytest.fixture()
def keypair(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAAtest bellhop@test")
    return str(key)


def _rest(handler) -> LambdaRest:
    rest = LambdaRest(api_key="k")
    rest._client = httpx.AsyncClient(
        base_url="https://cloud.lambda.ai/api/v1",
        transport=httpx.MockTransport(handler),
    )
    return rest


def _json(data, status=200):
    return httpx.Response(status, json={"data": data})


def _err(code, message, status=400):
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


# --- gpu= vocabulary ----------------------------------------------------------

def test_gpu_alias_crosses_with_count():
    assert LambdaConfig(gpu="H100").resolve_instance_types() == [
        "gpu_1x_h100_sxm5", "gpu_1x_h100_pcie"]
    assert LambdaConfig(gpu="H100", gpu_count=8).resolve_instance_types() == [
        "gpu_8x_h100_sxm5", "gpu_8x_h100_pcie"]
    # normalization: case / dashes / spaces don't matter
    assert (LambdaConfig(gpu="h-100").resolve_instance_types()
            == LambdaConfig(gpu="H100").resolve_instance_types())


def test_gpu_a100_means_80gb_like_runpod():
    # RunPod's "A100" alias is 80GB-only; silently landing on 40GB cards
    # would OOM a job mid-training, so the Lambda alias matches.
    assert LambdaConfig(gpu="A100", gpu_count=8).resolve_instance_types() == [
        "gpu_8x_a100_80gb_sxm4"]
    assert LambdaConfig(gpu="A100-40GB").resolve_instance_types() == [
        "gpu_1x_a100_sxm4", "gpu_1x_a100"]


def test_gpu_verbatim_type_name_passes():
    assert LambdaConfig(gpu="gpu_8x_h100_sxm5").resolve_instance_types() == ["gpu_8x_h100_sxm5"]
    assert LambdaConfig(instance_type="gpu_1x_gh200").resolve_instance_types() == ["gpu_1x_gh200"]


def test_gpu_and_instance_type_both_set_raises():
    with pytest.raises(PreflightError, match="not both"):
        LambdaConfig(gpu="H100", instance_type="gpu_1x_h100_pcie").resolve_instance_types()


def test_gpu_required_lambda_is_gpu_only():
    with pytest.raises(PreflightError, match="GPU-only"):
        LambdaConfig().resolve_instance_types()


def test_gpu_unknown_short_name_raises():
    with pytest.raises(PreflightError, match="known aliases"):
        LambdaConfig(gpu="Z9000").resolve_instance_types()


# --- timeouts scale with boot time ---------------------------------------------

def test_provision_timeout_widens_for_8x():
    assert LambdaConfig(gpu="H100").provision_timeout == timedelta(seconds=900)
    assert LambdaConfig(gpu="H100", gpu_count=8).provision_timeout == timedelta(seconds=1800)
    # the 8x marker is honored in verbatim names too
    assert LambdaConfig(gpu="gpu_8x_h100_sxm5").provision_timeout == timedelta(seconds=1800)


def test_timeouts_explicit_win_and_survive_replace():
    cfg = LambdaConfig(gpu="H100", gpu_count=8, provision_timeout=timedelta(seconds=60))
    assert cfg.provision_timeout == timedelta(seconds=60)
    # run() re-names the config per slug via dataclasses.replace
    assert replace(cfg, name="x").provision_timeout == timedelta(seconds=60)
    assert replace(cfg, name="x").resolve_instance_types() == cfg.resolve_instance_types()


# --- launch body ---------------------------------------------------------------

def test_launch_body_shape_and_name_stamp():
    body = LambdaConfig(gpu="H100", name="bellhop-demo").to_launch_body(
        "gpu_1x_h100_pcie", "us-east-1", "bellhop-abc")
    assert body["region_name"] == "us-east-1"
    assert body["instance_type_name"] == "gpu_1x_h100_pcie"
    assert body["ssh_key_names"] == ["bellhop-abc"]     # API: exactly one
    assert _stamp_epoch(body["name"])                   # gc's age signal
    assert "image" not in body and "user_data" not in body


def test_launch_body_image_spellings():
    assert (LambdaConfig(gpu="H100", image="ubuntu-lts")
            .to_launch_body("t", "r", "k")["image"] == {"family": "ubuntu-lts"})
    assert (LambdaConfig(gpu="H100", image={"id": "img-123"})
            .to_launch_body("t", "r", "k")["image"] == {"id": "img-123"})


# --- ssh key registration --------------------------------------------------------

def test_pubkey_blob_ignores_comment():
    assert _pubkey_blob("ssh-ed25519 AAAAtest me@laptop") == _pubkey_blob("ssh-ed25519 AAAAtest")


def test_ensure_ssh_key_matches_existing_by_blob(keypair):
    async def handler(req):
        assert req.url.path.endswith("/ssh-keys") and req.method == "GET"
        return _json([{"id": "1", "name": "old-name", "public_key": "ssh-ed25519 AAAAtest other@host"}])

    name = asyncio.run(_ensure_ssh_key(_rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair)))
    assert name == "old-name"


def test_ensure_ssh_key_registers_when_absent(keypair):
    posted = {}

    async def handler(req):
        if req.method == "GET":
            return _json([])
        posted.update(json.loads(req.content))
        return _json({"id": "2", "name": posted["name"], "public_key": posted["public_key"]})

    name = asyncio.run(_ensure_ssh_key(_rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair)))
    assert name.startswith("bellhop-") and posted["public_key"].startswith("ssh-ed25519")


def test_ensure_ssh_key_lost_registration_race_relists(keypair):
    calls = {"gets": 0}

    async def handler(req):
        if req.method == "GET":
            calls["gets"] += 1
            if calls["gets"] == 1:
                return _json([])  # nothing yet -> we try to register
            # second list: the racing process's registration is visible now
            return _json([{"id": "1", "name": "their-name", "public_key": "ssh-ed25519 AAAAtest x"}])
        return _err("global/duplicate", "name already in use")

    name = asyncio.run(_ensure_ssh_key(_rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair)))
    assert name == "their-name"


def test_ensure_ssh_key_preregistered_name_must_exist(keypair):
    async def handler(req):
        return _json([{"id": "1", "name": "real", "public_key": "ssh-ed25519 AAAA x"}])

    with pytest.raises(PreflightError, match="not registered"):
        asyncio.run(_ensure_ssh_key(
            _rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair, ssh_key_name="typo")))


# --- launch: candidates x regions against the live catalog ------------------------

_CATALOG = {
    "gpu_1x_h100_sxm5": {"instance_type": {"name": "gpu_1x_h100_sxm5"},
                         "regions_with_capacity_available": []},
    "gpu_1x_h100_pcie": {"instance_type": {"name": "gpu_1x_h100_pcie"},
                         "regions_with_capacity_available": [
                             {"name": "us-east-1"}, {"name": "us-west-1"}]},
}


def test_launch_walks_candidates_and_returns_id(keypair):
    launched = []

    async def handler(req):
        if req.url.path.endswith("/instance-types"):
            return _json(_CATALOG)
        launched.append(json.loads(req.content))
        return _json({"instance_ids": ["abc123"]})

    iid = asyncio.run(_launch(_rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair), "k"))
    assert iid == "abc123"
    # sxm5 had no capacity anywhere -> skipped; pcie tried in its first live region
    assert launched[0]["instance_type_name"] == "gpu_1x_h100_pcie"
    assert launched[0]["region_name"] == "us-east-1"


def test_launch_capacity_race_falls_through_regions(keypair):
    launched = []

    async def handler(req):
        if req.url.path.endswith("/instance-types"):
            return _json(_CATALOG)
        body = json.loads(req.content)
        launched.append(body["region_name"])
        if body["region_name"] == "us-east-1":  # catalog said yes, launch says no (TOCTOU)
            return _err("instance-operations/launch/insufficient-capacity",
                        "Not enough capacity to fulfill launch request.")
        return _json({"instance_ids": ["xyz"]})

    iid = asyncio.run(_launch(_rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair), "k"))
    assert iid == "xyz" and launched == ["us-east-1", "us-west-1"]


def test_launch_no_capacity_anywhere_is_capacity_error(keypair):
    dry = {t: {**v, "regions_with_capacity_available": []} for t, v in _CATALOG.items()}

    async def handler(req):
        return _json(dry)

    with pytest.raises(ProvisionError) as ei:
        asyncio.run(_launch(_rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair), "k"))
    assert is_capacity_error(ei.value)


def test_launch_pinned_region_attempted_despite_dry_catalog(keypair):
    launched = []

    async def handler(req):
        if req.url.path.endswith("/instance-types"):
            return _json(_CATALOG)
        launched.append(json.loads(req.content)["region_name"])
        return _json({"instance_ids": ["ok"]})

    # sxm5 shows no capacity, but a pinned region is launch-verified anyway
    cfg = LambdaConfig(gpu="H100", region="europe-central-1", ssh_key=keypair)
    assert asyncio.run(_launch(_rest(handler), cfg, "k")) == "ok"
    assert launched == ["europe-central-1"]


def test_launch_unknown_type_raises_preflight(keypair):
    async def handler(req):
        return _json(_CATALOG)

    with pytest.raises(PreflightError, match="no Lambda instance type matches"):
        asyncio.run(_launch(_rest(handler), LambdaConfig(gpu="gpu_1x_madeup", ssh_key=keypair), "k"))


def test_launch_error_carries_lambda_error_code(keypair):
    # is_capacity_error must see the structured code, not just message prose
    async def handler(req):
        if req.url.path.endswith("/instance-types"):
            return _json(_CATALOG)
        return _err("instance-operations/launch/insufficient-capacity", "different prose")

    with pytest.raises(ProvisionError) as ei:
        asyncio.run(_launch(_rest(handler), LambdaConfig(gpu="H100", ssh_key=keypair), "k"))
    assert is_capacity_error(ei.value)


# --- instance surface -------------------------------------------------------------

def test_instance_direct_networking(keypair):
    inst = LambdaInstance(LambdaRest(api_key="k"), "i-1", LambdaConfig(gpu="H100", ssh_key=keypair))
    inst._meta = {"ip": "1.2.3.4", "status": "active"}
    assert inst.host == "1.2.3.4"
    assert inst.mapped_port(22) == 22 and inst.mapped_port(8000) == 8000  # no NAT
    host, port = inst._ssh_endpoint()
    assert (host, port) == ("1.2.3.4", 22)
    assert "ubuntu@1.2.3.4" in " ".join(inst._ssh_argv())


def test_exec_default_timeout_is_unbounded():
    import inspect

    from bellhop.nebius_box import NebiusVm

    for cls in (LambdaInstance, NebiusVm):
        assert inspect.signature(cls.exec).parameters["timeout"].default is None


# --- gc: the only backstop a TTL-less provider gets --------------------------------

def test_stamp_epoch_parse():
    assert _stamp_epoch("bellhop-demo-t1756219000") == 1756219000
    assert _stamp_epoch("bellhop-demo") is None          # unstamped: never reaped
    assert _stamp_epoch("someone-else-t1756219000") is None  # not ours: never reaped


def test_launch_name_forced_reapable():
    # a user-chosen name must not make the instance invisible to gc
    body = LambdaConfig(gpu="H100", name="myexp").to_launch_body("t", "r", "k")
    assert body["name"].startswith("bellhop-myexp")
    assert _stamp_epoch(body["name"]) is not None


def test_gc_reaps_only_stamped_and_old(monkeypatch):
    import time as _time

    now = int(_time.time())
    instances = [
        {"id": "old", "name": f"bellhop-x-t{now - 7200}", "status": "active"},
        {"id": "new", "name": f"bellhop-y-t{now - 60}", "status": "active"},
        {"id": "keeper", "name": "my-hand-made-box", "status": "active"},
        {"id": "gone", "name": f"bellhop-z-t{now - 7200}", "status": "terminated"},
    ]
    killed = []

    async def handler(req):
        if req.method == "GET":
            return _json(instances)
        killed.extend(json.loads(req.content)["instance_ids"])
        return _json({"terminated_instances": []})

    rest = _rest(handler)
    monkeypatch.setattr("bellhop.lambda_box.LambdaRest", lambda api_key=None: rest)
    reaped = asyncio.run(gc_instances(timedelta(hours=1)))
    assert [r["id"] for r in reaped] == ["old"] and killed == ["old"]
    assert reaped[0]["age_hours"] >= 2.0


def test_gc_dry_run_terminates_nothing(monkeypatch):
    import time as _time

    async def handler(req):
        assert req.method == "GET", "dry_run must not POST terminate"
        return _json([{"id": "old", "name": f"bellhop-t{int(_time.time()) - 7200}",
                       "status": "active"}])

    rest = _rest(handler)
    monkeypatch.setattr("bellhop.lambda_box.LambdaRest", lambda api_key=None: rest)
    reaped = asyncio.run(gc_instances(timedelta(hours=1), dry_run=True))
    assert len(reaped) == 1


# --- rest client behavior ----------------------------------------------------------

def test_rest_429_retries_with_backoff(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    calls = {"n": 0}

    async def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return _json([])

    assert asyncio.run(_rest(handler).list_instances()) == []
    assert calls["n"] == 3 and sleeps == [10.0, 20.0]


def test_rest_error_envelope_shows_code_message_suggestion():
    async def handler(req):
        return httpx.Response(400, json={"error": {
            "code": "global/invalid-parameters", "message": "bad", "suggestion": "fix it"}})

    with pytest.raises(ProvisionError, match=r"global/invalid-parameters: bad \(fix it\)"):
        asyncio.run(_rest(handler).launch({}))


def test_terminate_404_is_idempotent():
    async def handler(req):
        return _err("global/object-does-not-exist", "gone", status=404)

    asyncio.run(_rest(handler).terminate(["dead"]))  # must not raise


def test_missing_api_key_raises(monkeypatch):
    from bellhop.lambda_box import _api_key

    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    with pytest.raises(PreflightError, match="LAMBDA_API_KEY"):
        _api_key(None)


# --- watchdog: the client-side max_lifetime ------------------------------------------

def test_lifetime_watchdog_flags_and_tears_down():
    from bellhop.sshbox import lifetime_watchdog

    class Box:
        id = "i-1"
        _noun = "instance"
        _lifetime_expired = False
        torn_down = False

        async def teardown(self):
            self.torn_down = True

    box = Box()
    asyncio.run(lifetime_watchdog(box, timedelta(seconds=0.01)))
    assert box._lifetime_expired and box.torn_down
