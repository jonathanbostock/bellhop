"""Exception hierarchy.

Mirrors the exit-code ladder of the original ``run.sh`` driver so callers can
branch on failure mode instead of parsing exit codes:

    10 preflight, 20 provision, 30 never-ready,
    40 remote-job-failed, 41 exec-timeout, 42 remote-call-raised,
    50 results-missing, 60 gcs-upload-failed.

The hierarchy is provider-agnostic (RunPod, Modal, Lambda, Nebius):
``ProvisionError`` is raised when any backend's box fails to come up,
``RemoteJobError`` when the job exits non-zero on any of them, and so on.
"""

from __future__ import annotations


class BellhopError(Exception):
    """Base for everything this library raises."""

    exit_code = 1


# Back-compat alias: the base used to be RunPod-specific.
RunpodError = BellhopError


class PreflightError(BellhopError):
    """Bad config or missing local prerequisite (key, gcloud, codebase, modal)."""

    exit_code = 10


class ProvisionError(BellhopError):
    """Box create failed (pod out of stock / bad image-gpu id, or sandbox create)."""

    exit_code = 20


# The providers' ways of saying "no stock", collected from live runs (issue
# #27's probe matrix and stock-outs since) and from each API's documented
# error codes. Heuristic by necessity — RunPod gives prose, not codes — so
# keep entries lowercase substrings.
CAPACITY_SIGNATURES = (
    # RunPod
    "no capacity",                       # graphql null-pod spelling
    "does not have the resources",       # graphql machine-match failure
    "no longer any instances available", # rest out-of-stock
    "out of stock",
    "no instances",
    "insufficient resources",            # createCluster out-of-stock (M0 probe)
    # Lambda (documented code instance-operations/launch/insufficient-capacity,
    # default message "Not enough capacity to fulfill launch request.")
    "insufficient-capacity",
    "not enough capacity",
    # Nebius (documented "Not enough resources"; gRPC RESOURCE_EXHAUSTED also
    # covers quota — retrying elsewhere is the right move for both)
    "not enough resources",
    "resource_exhausted",
    "resource exhausted",
    "quota limit exceeded",
)


def is_capacity_error(err: BaseException) -> bool:
    """Best-effort: does this provision failure look like a stock-out?

    Lets callers (retry loops, live test suites) separate "the provider has no
    machines right now" from "my request is broken". False negatives are
    possible when a provider invents new prose; treat a True as reliable and a
    False as "unknown".
    """
    msg = str(err).lower()
    return any(sig in msg for sig in CAPACITY_SIGNATURES)


class PodNotReadyError(BellhopError):
    """Box never became functional within the timeout."""

    exit_code = 30


class RemoteJobError(BellhopError):
    """The remote command(s) exited non-zero."""

    exit_code = 40

    def __init__(self, message: str, *, remote_exit: int, log_tail: str = ""):
        super().__init__(message)
        self.remote_exit = remote_exit
        self.log_tail = log_tail


class ExecTimeoutError(BellhopError):
    """An ``exec()``'s client-side ``timeout=`` expired.

    Only raised when a caller opted into a finite timeout (the default is
    unbounded — the box's server-side TTL is the backstop). NB the remote
    process may still be running on the box; only the local wait was killed.
    """

    exit_code = 41


class RemoteCallError(BellhopError):
    """A ``call()``'d function raised on the box.

    Carries the remote traceback text. When the remote exception object could
    itself be unpickled, *that* exception is re-raised locally with this error
    as its ``__cause__`` — so callers catch the original type and reach the
    remote traceback via ``e.__cause__.remote_traceback``.
    """

    exit_code = 42

    def __init__(self, message: str, *, remote_traceback: str = ""):
        super().__init__(message)
        self.remote_traceback = remote_traceback


class ResultsMissingError(BellhopError):
    """The job produced no results directory to pull back."""

    exit_code = 50


class GcsUploadError(BellhopError):
    """Uploading the pulled artifacts to GCS failed."""

    exit_code = 60
