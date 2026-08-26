"""bellhop: check your code into an ephemeral box (RunPod pod or Modal sandbox), run it, bring results back, check out."""

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
from .modal_box import ModalConfig, Sandbox, sandbox
from .pod import GPU_ALIASES, IMAGE_PRESETS, Pod, PodConfig, pod
from .probes import HttpProbe, LogMarkerProbe, ReadyProbe, SshProbe, TcpProbe
from .rest import RunpodRest
from .run import RunResult, RunSpec, run, run_many

__all__ = [
    # backend-agnostic surface
    "run", "run_many", "RunSpec", "RunResult",
    "open_box", "ExecBox", "ExecResult", "call",
    # RunPod backend
    "pod", "Pod", "PodConfig", "IMAGE_PRESETS", "GPU_ALIASES",
    "RunpodRest", "RunpodGraphQL",
    # RunPod Instant Clusters (multi-node)
    "cluster", "Cluster", "ClusterConfig", "ClusterJobError",
    "run_cluster", "list_clusters", "gc_clusters",
    "ReadyProbe", "SshProbe", "TcpProbe", "HttpProbe", "LogMarkerProbe",
    # Modal backend
    "sandbox", "Sandbox", "ModalConfig",
    # errors
    "BellhopError", "RunpodError", "PreflightError", "ProvisionError", "PodNotReadyError",
    "RemoteJobError", "ExecTimeoutError", "RemoteCallError", "ResultsMissingError",
    "GcsUploadError", "is_capacity_error",
]
