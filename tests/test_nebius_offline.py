"""Offline unit tests for the Nebius backend — no live VM, no cost.

Config resolution / cloud-init / naming are pure logic (no SDK needed). The
SDK-touching tests use importorskip and double as the *import-surface canary*:
the `nebius` package is pre-1.0 and proto-generated, so if a version bump
reshapes the API, these fail in dev/CI instead of at a user's provision time.
"""

import re
from dataclasses import replace
from datetime import timedelta

import pytest

from bellhop import NebiusConfig, PreflightError
from bellhop.nebius_box import _NAME_STAMP, _safe_name, _stamp_epoch


@pytest.fixture()
def keypair(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAAtest bellhop@test")
    return str(key)


# --- platform / preset vocabulary ------------------------------------------------

def test_gpu_alias_resolves_platform():
    assert NebiusConfig(gpu="H100").resolve_platform() == "gpu-h100-sxm"
    assert NebiusConfig(gpu="h-200").resolve_platform() == "gpu-h200-sxm"  # normalized
    assert NebiusConfig().resolve_platform() == "cpu-d3"                   # no gpu -> CPU box


def test_gpu_verbatim_platform_passes():
    assert NebiusConfig(gpu="gpu-h100-sxm").resolve_platform() == "gpu-h100-sxm"
    assert NebiusConfig(platform="gpu-l40s-d").resolve_platform() == "gpu-l40s-d"


def test_gpu_and_platform_both_set_raises():
    with pytest.raises(PreflightError, match="not both"):
        NebiusConfig(gpu="H100", platform="gpu-h100-sxm").resolve_platform()


def test_gpu_unknown_short_name_raises():
    with pytest.raises(PreflightError, match="known aliases"):
        NebiusConfig(gpu="Z9000").resolve_platform()


def test_preset_derived_from_gpu_count():
    assert NebiusConfig(gpu="H100").resolve_preset() == "1gpu-16vcpu-200gb"
    assert NebiusConfig(gpu="H200", gpu_count=8).resolve_preset() == "8gpu-128vcpu-1600gb"
    assert NebiusConfig().resolve_preset() == "4vcpu-16gb"


def test_preset_explicit_wins():
    assert NebiusConfig(gpu="H100", preset="1gpu-20vcpu-200gb").resolve_preset() == "1gpu-20vcpu-200gb"


def test_preset_required_when_no_default_shape():
    # B200/L40S shapes aren't pinned down -> demand an explicit preset rather
    # than guess one the API would reject after the disk was validated.
    with pytest.raises(PreflightError, match="set preset="):
        NebiusConfig(gpu="B200").resolve_preset()
    with pytest.raises(PreflightError, match="set preset="):
        NebiusConfig(gpu="H100", gpu_count=4).resolve_preset()


def test_image_family_defaults_follow_compute():
    assert NebiusConfig(gpu="H100").resolve_image_family() == "ubuntu24.04-cuda12"
    assert NebiusConfig().resolve_image_family() == "ubuntu24.04-driverless"
    assert NebiusConfig(image_family="ubuntu24.04-cuda13.0").resolve_image_family() == "ubuntu24.04-cuda13.0"


def test_config_survives_replace():
    cfg = NebiusConfig(gpu="H100", gpu_count=8, max_lifetime=timedelta(hours=8))
    assert replace(cfg, name="x").resolve_preset() == "8gpu-128vcpu-1600gb"
    assert replace(cfg, name="x").max_lifetime == timedelta(hours=8)


# --- project / auth preflights -----------------------------------------------------

def test_project_id_from_env(monkeypatch):
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "project-e00env")
    assert NebiusConfig().resolve_project_id() == "project-e00env"
    assert NebiusConfig(project_id="project-e00x").resolve_project_id() == "project-e00x"


def test_project_id_missing_raises(monkeypatch):
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    with pytest.raises(PreflightError, match="NEBIUS_PROJECT_ID"):
        NebiusConfig().resolve_project_id()


# --- cloud-init: the images have no default user -----------------------------------

def test_cloud_init_creates_user_with_key(keypair):
    ci = NebiusConfig(ssh_key=keypair).cloud_init()
    assert ci.startswith("#cloud-config")
    assert "- name: ubuntu" in ci
    assert "sudo: ALL=(ALL) NOPASSWD:ALL" in ci
    assert "ssh-ed25519 AAAAtest" in ci


def test_cloud_init_rejects_reserved_users(keypair):
    for user in ("root", "admin"):
        with pytest.raises(PreflightError, match="reserve"):
            NebiusConfig(ssh_key=keypair, ssh_user=user).cloud_init()


# --- naming: sanitized for Nebius, stamped for gc -----------------------------------

def test_safe_name_sanitizes():
    assert _safe_name("bellhop-My_Sweep.01") == "bellhop-my-sweep-01"
    assert _safe_name("///") == "bellhop"


def test_stamped_name_parses_back():
    name = NebiusConfig(name="bellhop-demo").stamped_name()
    assert _NAME_STAMP.match(name)
    assert _stamp_epoch(name) is not None
    assert _stamp_epoch("bellhop-demo") is None        # unstamped: never reaped
    assert _stamp_epoch("their-vm-t1756219000") is None  # not ours: never reaped


def test_stamped_name_bounds_length():
    name = NebiusConfig(name="bellhop-" + "x" * 100).stamped_name()
    assert len(name) <= 63 and _NAME_STAMP.match(name)


# --- SDK import surface (the version-drift canary) ----------------------------------

def test_sdk_import_surface():
    pytest.importorskip("nebius")
    from bellhop.nebius_box import _import_nebius

    nb = _import_nebius()
    for sym in ("SDK", "InstanceServiceClient", "CreateInstanceRequest",
                "GetInstanceRequest", "DeleteInstanceRequest", "ListInstancesRequest",
                "InstanceSpec", "ResourcesSpec", "AttachedDiskSpec", "ManagedDisk",
                "DiskSpec", "SourceImageFamily", "NetworkInterfaceSpec", "IPAddress",
                "PublicIPAddress", "InstanceStatus", "ResourceMetadata",
                "SubnetServiceClient", "ListSubnetsRequest", "RequestError"):
        assert hasattr(nb, sym), f"nebius SDK no longer exposes {sym}"
    # the states the provision loop branches on
    states = nb.InstanceStatus.InstanceState
    for s in ("CREATING", "STARTING", "RUNNING", "STOPPED", "DELETING", "ERROR"):
        assert hasattr(states, s)


def test_create_request_shape(keypair, monkeypatch):
    pytest.importorskip("nebius")
    from bellhop.nebius_box import _create_request, _import_nebius

    nb = _import_nebius()
    cfg = NebiusConfig(gpu="H100", gpu_count=8, ssh_key=keypair, disk_gb=200,
                       name="bellhop-demo")
    req = _create_request(nb, cfg, "project-e00x", "subnet-e00y")
    assert req.metadata.parent_id == "project-e00x"
    assert _NAME_STAMP.match(req.metadata.name)
    assert req.spec.resources.platform == "gpu-h100-sxm"
    assert req.spec.resources.preset == "8gpu-128vcpu-1600gb"
    disk = req.spec.boot_disk.managed_disk
    assert disk.spec.size_gibibytes == 200
    assert disk.spec.source_image_family.image_family == "ubuntu24.04-cuda12"
    nic = req.spec.network_interfaces[0]
    assert nic.subnet_id == "subnet-e00y"
    assert nic.public_ip_address is not None
    assert "ssh-ed25519 AAAAtest" in req.spec.cloud_init_user_data


def test_vm_state_and_host_parsing(keypair):
    pytest.importorskip("nebius")
    from bellhop.nebius_box import NebiusVm, _import_nebius

    nb = _import_nebius()
    box = NebiusVm(sdk=None, nb=nb, vm_id="computeinstance-e00z",
                   config=NebiusConfig(ssh_key=keypair))
    assert box.state == "UNSPECIFIED" and box.host is None  # nothing fetched yet

    class _Nic:
        class public_ip_address:
            address = "1.2.3.4/32"

    class _Status:
        state = nb.InstanceStatus.InstanceState.RUNNING
        reconciling = False
        network_interfaces = [_Nic()]

    class _Inst:
        status = _Status()

    box._inst = _Inst()
    assert box.state == "RUNNING"
    assert box.host == "1.2.3.4"  # CIDR suffix stripped
    assert box.mapped_port(22) == 22 and box.mapped_port(8000) == 8000
    assert "ubuntu@1.2.3.4" in " ".join(box._ssh_argv())
