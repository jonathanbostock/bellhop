"""bellhop: check your code into an ephemeral box (RunPod pod, Modal sandbox, Lambda instance or Nebius VM), run it, bring results back, check out."""

from .backend import ExecBox, ExecResult, open_box
from .call import call
from .cluster import (
    Cluster,
    ClusterConfig,
    ClusterJobError,
    cluster,
    gc_clusters,
    list_clusters,
    run_cluster,
)
from .errors import (
    BellhopError,
    ExecTimeoutError,
    GcsUploadError,
    PodNotReadyError,
    PreflightError,
    ProvisionError,
    RemoteCallError,
    RemoteJobError,
    ResultsMissingError,
    RunpodError,
    is_capacity_error,
)
from .graphql import RunpodGraphQL
from .lambda_box import (
    LAMBDA_GPU_ALIASES,
    LambdaConfig,
    LambdaInstance,
    LambdaRest,
    gc_instances,
    instance,
    list_instances,
)
from .modal_box import ModalConfig, Sandbox, sandbox
from .nebius_box import (
    NEBIUS_GPU_PLATFORMS,
    NebiusConfig,
    NebiusVm,
    gc_vms,
    list_vms,
    vm,
)
from .pod import GPU_ALIASES, IMAGE_PRESETS, Pod, PodConfig, pod
from .probes import HttpProbe, LogMarkerProbe, ReadyProbe, SshProbe, TcpProbe
from .rest import RunpodRest
from .run import RunResult, RunSpec, run, run_many
from .sshbox import SshBox

__all__ = [
    # backend-agnostic surface
    "run", "run_many", "RunSpec", "RunResult",
    "open_box", "ExecBox", "ExecResult", "SshBox", "call",
    # RunPod backend
    "pod", "Pod", "PodConfig", "IMAGE_PRESETS", "GPU_ALIASES",
    "RunpodRest", "RunpodGraphQL",
    # RunPod Instant Clusters (multi-node)
    "cluster", "Cluster", "ClusterConfig", "ClusterJobError",
    "run_cluster", "list_clusters", "gc_clusters",
    "ReadyProbe", "SshProbe", "TcpProbe", "HttpProbe", "LogMarkerProbe",
    # Modal backend
    "sandbox", "Sandbox", "ModalConfig",
    # Lambda Cloud backend
    "instance", "LambdaInstance", "LambdaConfig", "LambdaRest",
    "LAMBDA_GPU_ALIASES", "list_instances", "gc_instances",
    # Nebius backend
    "vm", "NebiusVm", "NebiusConfig", "NEBIUS_GPU_PLATFORMS",
    "list_vms", "gc_vms",
    # errors
    "BellhopError", "RunpodError", "PreflightError", "ProvisionError", "PodNotReadyError",
    "RemoteJobError", "ExecTimeoutError", "RemoteCallError", "ResultsMissingError",
    "GcsUploadError", "is_capacity_error",
]
