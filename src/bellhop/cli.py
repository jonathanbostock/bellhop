"""CLI over run() — one codebase on an ephemeral RunPod pod or Modal sandbox."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import timedelta

from .errors import BellhopError
from .modal_box import ModalConfig
from .pod import PodConfig
from .run import DEFAULT_GCS_BASE, RunSpec, run


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bellhop", description="Run a codebase on an ephemeral RunPod pod or Modal sandbox.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="provision -> run -> pull -> GCS -> teardown")
    r.add_argument("--backend", choices=["runpod", "modal"], default="runpod")
    # shared
    r.add_argument("--slug", required=True)
    r.add_argument("--codebase", required=True, help="local dir OR git URL")
    r.add_argument("--run", required=True, help="the command(s) to run")
    r.add_argument("--setup", default=None, help="deps, run before --run")
    r.add_argument("--image", default=None, help="image (RunPod docker ref / Modal registry ref)")
    r.add_argument("--image-preset", default=None, help="RunPod: cpu-base/pytorch-* ; Modal: debian-slim/pytorch-cuda")
    r.add_argument("--results-subdir", default="results")
    r.add_argument("--local-out", default=None)
    r.add_argument("--gcs-base", default=DEFAULT_GCS_BASE)
    r.add_argument("--no-gcs", action="store_true", help="skip GCS upload")
    r.add_argument("--env-json", default=None, help="JSON object of extra box env vars")
    r.add_argument("--keep-pod", action="store_true", help="leave the box up after the run")
    r.add_argument("--gpu", default=None,
                   help="GPU short name, e.g. 'A100', 'H100', 'L4' (both backends; omit for a CPU box). "
                        "RunPod also accepts a full gpuTypeId like 'NVIDIA GeForce RTX 4090'.")
    r.add_argument("--max-lifetime-hours", type=float, default=None,
                   help="hard max box lifetime (both backends; default: config default)")
    r.add_argument("--run-timeout-hours", type=float, default=None,
                   help="client-side cap on the job step (default: none — the box TTL is the bound)")
    # RunPod-specific
    r.add_argument("--compute", choices=["cpu", "gpu"], default=None, help="RunPod: derived from --gpu when omitted")
    r.add_argument("--gpu-id", default=None, help="[deprecated] verbatim RunPod gpuTypeId; use --gpu")
    r.add_argument("--container-disk-gb", type=int, default=20)
    r.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="COMMUNITY")
    r.add_argument("--ready-timeout", type=int, default=420)
    # Modal-specific
    r.add_argument("--pip", action="append", default=None, help="Modal: pip-install onto the image (repeatable)")
    r.add_argument("--timeout-hours", type=float, default=None, help="[deprecated] use --max-lifetime-hours")

    # Instant Clusters have no server-side TTL, so leaks are on us to find
    c = sub.add_parser("clusters", help="list / reap RunPod Instant Clusters")
    csub = c.add_subparsers(dest="clusters_cmd", required=True)
    csub.add_parser("list", help="list the account's clusters")
    g = csub.add_parser("gc", help="delete clusters older than a threshold")
    g.add_argument("--older-than-hours", type=float, default=24.0)
    g.add_argument("--dry-run", action="store_true", help="report only, delete nothing")
    return p


def _build_backend(args, env: dict):
    max_lifetime = None
    if args.max_lifetime_hours is not None:
        max_lifetime = timedelta(hours=args.max_lifetime_hours)
    elif args.timeout_hours is not None:
        print("warning: --timeout-hours is deprecated; use --max-lifetime-hours", file=sys.stderr)
        max_lifetime = timedelta(hours=args.timeout_hours)
    if args.gpu_id:
        print("warning: --gpu-id is deprecated; use --gpu", file=sys.stderr)

    if args.backend == "modal":
        return ModalConfig(
            gpu=args.gpu,
            image=args.image,
            image_preset=args.image_preset,
            pip=list(args.pip or []),
            env=dict(env),
            max_lifetime=max_lifetime,
        )
    return PodConfig(
        compute=args.compute,
        gpu=args.gpu,
        gpu_id=args.gpu_id,
        image=args.image,
        image_preset=args.image_preset,
        container_disk_gb=args.container_disk_gb,
        cloud=args.cloud,
        env=dict(env),
        ready_timeout=timedelta(seconds=args.ready_timeout),
        max_lifetime=max_lifetime,
    )


def _clusters_main(args) -> int:
    from .cluster import gc_clusters, list_clusters

    try:
        if args.clusters_cmd == "list":
            for clu in asyncio.run(list_clusters()):
                print(f"{clu['id']}  {clu['gpuTypeId']} x{clu['gpuCountPerPod']}/node "
                      f"x{clu['podCount']} nodes  created {clu['createdAt']}  ({clu['name']})")
            return 0
        reaped = asyncio.run(gc_clusters(timedelta(hours=args.older_than_hours),
                                         dry_run=args.dry_run))
        verb = "would reap" if args.dry_run else "reaped"
        for clu in reaped:
            print(f"{verb} {clu['id']}  {clu['gpuTypeId']} x{clu['podCount']} nodes  "
                  f"age {clu['age_hours']}h")
        if not reaped:
            print(f"no clusters older than {args.older_than_hours}h")
        return 0
    except BellhopError as e:
        print(f"ERROR [{type(e).__name__}]: {e}", file=sys.stderr)
        return e.exit_code


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "clusters":
        return _clusters_main(args)
    env = json.loads(args.env_json) if args.env_json else {}

    backend = _build_backend(args, env)
    spec = RunSpec(
        slug=args.slug,
        codebase=args.codebase,
        run=args.run,
        setup=args.setup,
        results_subdir=args.results_subdir,
        local_out=args.local_out,
        gcs_base=None if args.no_gcs else args.gcs_base,
        env=dict(env),
        timeout=args.run_timeout_hours * 3600 if args.run_timeout_hours else None,
    )

    try:
        res = asyncio.run(run(spec, backend, keep_pod=args.keep_pod))
    except BellhopError as e:
        print(f"ERROR [{type(e).__name__}]: {e}", file=sys.stderr)
        return e.exit_code

    print("\n===================== BELLHOP RESULT =====================")
    print(f"backend:       {args.backend}")
    print(f"slug:          {res.slug}")
    print(f"box_id:        {res.pod_id} (torn down: {'no' if args.keep_pod else 'yes'})")
    print(f"remote_exit:   {res.remote_exit}")
    print(f"local_results: {res.local_results}")
    print(f"gcs_artifacts: {res.gcs_uri}")
    if res.retrieve_cmd:
        print(f"retrieve:      {res.retrieve_cmd}")
    print("-------------------- run.log (tail) --------------------")
    print(res.log_tail or "(no run.log)")
    print("=======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
