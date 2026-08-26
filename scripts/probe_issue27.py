"""Live probe matrix for issue #27 — which GraphQL create input 500s?

Reported: podFindAndDeployOnDemand returns "Something went wrong" for
custom image + docker_start_cmd (+ TTL), while the same config via REST
(no TTL) creates fine. This fires the mutation with each input variant
and deletes any pod it manages to create, so the whole run costs cents.

Run:  set -a; source ~/.env; set +a
      uv run --package bellhop-py python packages/bellhop/scripts/probe_issue27.py
"""

import asyncio
import shlex
from datetime import datetime, timedelta, timezone

from bellhop.errors import ProvisionError
from bellhop.graphql import RunpodGraphQL
from bellhop.rest import RunpodRest

RUNPOD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CUSTOM_IMAGE = "ubuntu:22.04"

# the docstring example from PodConfig.docker_start_cmd — what scimt-style
# sshd bootstraps look like (embedded quotes exercise the shlex path)
SSHD_BOOTSTRAP = (
    'apt-get update && apt-get install -y openssh-server && mkdir -p /run/sshd ~/.ssh '
    '&& echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys '
    '&& /usr/sbin/sshd -D'
)


# (gpuTypeId, cloudType) fallback ladder — "does not have the resources" /
# null just means no stock for that pair, so walk until one takes
CANDIDATES = [
    ("NVIDIA GeForce RTX 4090", "COMMUNITY"),
    ("NVIDIA GeForce RTX 4090", "SECURE"),
    ("NVIDIA RTX A4000", "COMMUNITY"),
    ("NVIDIA RTX A4000", "SECURE"),
    ("NVIDIA RTX A5000", "SECURE"),
]


def base(image: str) -> dict:
    return {
        "cloudType": "COMMUNITY",
        "name": "bellhop-probe27",
        "imageName": image,
        "gpuTypeId": "NVIDIA GeForce RTX 4090",
        "gpuCount": 1,
        "minVcpuCount": 1,      # runpodctl sends these; without them the
        "minMemoryInGb": 1,     # scheduler can pick an unfittable machine
        "containerDiskInGb": 10,
        "ports": "22/tcp",
        "env": [{"key": "PUBLIC_KEY", "value": "ssh-ed25519 AAAA probe"}],
    }


def ttl(inp: dict) -> dict:
    # server-side timers double as cleanup insurance if delete_pod fails
    now = datetime.now(timezone.utc)
    inp["stopAfter"] = (now + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    inp["terminateAfter"] = (now + timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return inp


def args(inp: dict, cmd: str) -> dict:
    inp["dockerArgs"] = f"bash -c {shlex.quote(cmd)}"  # pod.py's exact spelling
    return inp


CASES = [
    ("A: base image + ttl (baseline)",       ttl(base(RUNPOD_IMAGE))),
    ("B: base image + args(simple) + ttl",   ttl(args(base(RUNPOD_IMAGE), "sleep infinity"))),
    ("C: custom image + ttl",                ttl(base(CUSTOM_IMAGE))),
    ("D: custom image + args(simple) + ttl", ttl(args(base(CUSTOM_IMAGE), "sleep infinity"))),
    ("E: custom image + args(sshd) + ttl",   ttl(args(base(CUSTOM_IMAGE), SSHD_BOOTSTRAP))),
    ("F: custom image + args(sshd), no ttl", args(base(CUSTOM_IMAGE), SSHD_BOOTSTRAP)),
    ("G: base image + args(sshd) + ttl",     ttl(args(base(RUNPOD_IMAGE), SSHD_BOOTSTRAP))),
]


CAPACITY = ("returned null", "does not have the resources", "no longer any instances")


async def try_case(gql: RunpodGraphQL, rest: RunpodRest, name: str, inp: dict) -> str:
    last = ""
    for gpu, cloud in CANDIDATES:
        inp["gpuTypeId"], inp["cloudType"] = gpu, cloud
        try:
            pod = await gql.create_pod_on_demand(inp)
        except ProvisionError as e:
            last = str(e)
            if any(s in last for s in CAPACITY):
                continue  # stock problem, not the bug — walk the ladder
            return f"FAIL  {last[:160]}"
        await rest.delete_pod(pod["id"])
        return f"OK    pod {pod['id']} created ({gpu}, {cloud}) and deleted"
    return f"FAIL(capacity-only)  {last[:120]}"


def scimt_exact() -> dict:
    """The failing build_flash_wheels.py config, via bellhop's real input builder."""
    from datetime import timedelta

    from bellhop.pod import PodConfig

    cfg = PodConfig(
        gpu="RTX 4090", gpu_count=1, container_disk_gb=60,
        image="nvidia/cuda:12.6.3-devel-ubuntu24.04",
        docker_start_cmd=SSHD_BOOTSTRAP,
        max_lifetime=timedelta(hours=3),
        name="flash-wheel-probe",
    )
    return cfg.to_graphql_input()


async def one_shot(gql: RunpodGraphQL, rest: RunpodRest, inp: dict, cloud: str) -> str:
    """No fallback ladder — probe exactly one (input, cloud) point."""
    inp = dict(inp, cloudType=cloud)
    try:
        pod = await gql.create_pod_on_demand(inp)
    except ProvisionError as e:
        return f"FAIL  {str(e)[:300]}"
    await rest.delete_pod(pod["id"])
    return f"OK    pod {pod['id']} created and deleted"


async def main() -> None:
    import sys

    async with RunpodGraphQL() as gql, RunpodRest() as rest:
        if "--full" in sys.argv:  # A-G answered the dockerArgs question; skip by default
            for name, inp in CASES:
                print(f"{name:40s} -> {await try_case(gql, rest, name, inp)}", flush=True)
        exact = scimt_exact()
        for name, inp, cloud in [
            ("H: scimt-exact, COMMUNITY", exact, "COMMUNITY"),
            ("I: scimt-exact, SECURE", exact, "SECURE"),
            ("J: scimt-exact minus args, COMMUNITY", {k: v for k, v in exact.items() if k != "dockerArgs"}, "COMMUNITY"),
            ("K: base image disk10 no args, COMMUNITY", ttl(base(RUNPOD_IMAGE)), "COMMUNITY"),
        ]:
            print(f"{name:40s} -> {await one_shot(gql, rest, inp, cloud)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
