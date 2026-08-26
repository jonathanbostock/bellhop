"""The Lambda Cloud backend — a plain SSH VM behind a small REST API.

Lambda Cloud (lambda.ai, née Lambda Labs) is the "boring" GPU provider:
on-demand instances with fixed per-type pricing, a documented REST API
(``cloud.lambda.ai/api/v1``), and a real per-region capacity signal — the
things RunPod's spot market trades away. The box itself is a stock Ubuntu VM
(Lambda Stack preinstalled) reached as ``ubuntu@ip:22``, so the transport is
:class:`bellhop.sshbox.SshBox` unchanged.

What this module adds on top of the transport:

- **Launch with live capacity data.** ``GET /instance-types`` reports every
  type's ``regions_with_capacity_available``; a ``gpu=`` alias expands to
  candidate type names, gets filtered against that catalog, and launch
  attempts walk (type, region) pairs in preference order — collecting every
  error, like the RunPod GraphQL path does. The pre-filter is an
  optimization only: launch remains the source of truth (a pinned region is
  attempted even when the catalog says it's dry — TOCTOU cuts both ways).
- **Documented rate limits, respected.** 1 request/second generally and one
  ``launch`` per 12 seconds — enforced by a *process-wide* pacer on
  :class:`LambdaRest` (plain monotonic timestamps at class level, no Lock: a
  Lock would bind the first event loop and break the next ``asyncio.run``),
  so a ``run_many()`` fan-out doesn't trip the limit. 429s are retried with
  backoff on top, belt and braces.
- **SSH key auto-registration.** Launch requires exactly one *pre-registered*
  key name; bellhop finds a registered key matching the local public key
  (type+blob compare, comment ignored) or registers it as
  ``bellhop-<sha256[:12]>``, racing registrations resolved by re-listing.
- **No server-side TTL exists** (confirmed against the full OpenAPI spec) —
  billing runs until terminate. ``max_lifetime=`` therefore arms an
  *in-process* watchdog (the ``cluster.py`` pattern) and the launch stamps a
  ``-t<epoch>`` suffix onto the instance name so ``bellhop lambda gc`` can
  reap anything a dead host left behind.

Auth is ``LAMBDA_API_KEY`` (Bearer). Errors follow the documented envelope
``{"error": {"code", "message", "suggestion"?}}`` — we branch on ``code`` and
surface ``code: message`` so :func:`bellhop.errors.is_capacity_error` can see
``instance-operations/launch/insufficient-capacity``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, AsyncIterator

import httpx

from .errors import (
    BellhopError,
    PodNotReadyError,
    PreflightError,
    ProvisionError,
)
from .pod import _canon_gpu
from .probes import ReadyProbe, SshProbe
from .sshbox import (
    SshBox,
    ensure_workspace,
    install_pip,
    lifetime_watchdog,
    pubkey_text,
    resolve_ssh_key,
    wait_ready,
)

DEFAULT_BASE = "https://cloud.lambda.ai/api/v1"

# Canonical GPU vocabulary — the same short names as pod.GPU_ALIASES, expanded
# to Lambda instance-type *variant* suffixes in preference order. A candidate
# type name is gpu_{count}x_{variant}; names that don't exist at the requested
# count simply drop out against the live catalog. NB "A100" means 80GB
# variants only (matching the RunPod alias) — say "A100-40GB" for the cheap
# cards, or pass a verbatim type name.
LAMBDA_GPU_ALIASES: dict[str, list[str]] = {
    "A10": ["a10"],
    "A100": ["a100_80gb_sxm4"],
    "A100-40GB": ["a100_sxm4", "a100"],
    "A100-80GB": ["a100_80gb_sxm4"],
    "A6000": ["a6000"],
    "B200": ["b200_sxm6"],
    "GH200": ["gh200"],
    "H100": ["h100_sxm5", "h100_pcie"],
    "RTX6000": ["rtx6000"],
    "V100": ["v100"],
}

_ALIAS_LOOKUP = {_canon_gpu(k): v for k, v in LAMBDA_GPU_ALIASES.items()}

# Launch attempts are expensive (12s pacing each); bound the worst case.
_MAX_LAUNCH_ATTEMPTS = 8

# The -t<epoch> name stamp: how `gc_instances` knows an instance is ours and
# how old it is (the Instance record carries no launch timestamp of its own).
_NAME_STAMP = re.compile(r"^bellhop.*-t(\d{9,12})$")


def _api_key(explicit: str | None) -> str:
    import os

    key = explicit or os.environ.get("LAMBDA_API_KEY")
    if not key:
        raise PreflightError("LAMBDA_API_KEY not set (pass api_key= or export it)")
    return key


class LambdaRest:
    """Async client over the Lambda Cloud API. Use as an async context manager.

    Deliberately tiny, like :class:`bellhop.rest.RunpodRest` — just the verbs
    the backend needs. Pacing state is *class*-level so every client in the
    process shares one budget (the documented limits are per account, and a
    sweep runs many clients).
    """

    # Documented: "one request per second … /instance-operations/launch …
    # one request per 12 seconds". Class attributes so tests can zero them.
    min_request_interval = 1.0
    min_launch_interval = 12.0
    _last_request = 0.0
    _last_launch = 0.0

    def __init__(self, api_key: str | None = None, base: str = DEFAULT_BASE, timeout: float = 60.0):
        self.base = base.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base,
            headers={"Authorization": f"Bearer {_api_key(api_key)}"},
            timeout=timeout,
        )

    async def __aenter__(self) -> "LambdaRest":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _pace(self, *, launch: bool) -> None:
        # No asyncio.Lock here: the check-and-set has no await between reading
        # the clock and stamping it, so it's atomic within one event loop, and
        # plain floats survive across asyncio.run() loops where a Lock can't.
        while True:
            now = time.monotonic()
            wait = LambdaRest._last_request + self.min_request_interval - now
            if launch:
                wait = max(wait, LambdaRest._last_launch + self.min_launch_interval - now)
            if wait <= 0:
                LambdaRest._last_request = now
                if launch:
                    LambdaRest._last_launch = now
                return
            await asyncio.sleep(wait)

    @staticmethod
    def _err_text(resp: httpx.Response) -> str:
        # Documented envelope: {"error": {"code", "message", "suggestion"?}}.
        # Keep the code in front — it's the stable part (messages drift), and
        # is_capacity_error matches on it.
        try:
            err = resp.json().get("error") or {}
        except Exception:
            return resp.text[:500]
        text = f"{err.get('code', 'unknown')}: {err.get('message', '')}"
        if err.get("suggestion"):
            text += f" ({err['suggestion']})"
        return text

    async def _request(self, method: str, path: str, json: dict | None = None) -> httpx.Response:
        launch = path == "/instance-operations/launch"
        backoff = 10.0
        for attempt in range(4):
            await self._pace(launch=launch)
            resp = await self._client.request(method, path, json=json)
            if resp.status_code != 429:
                return resp
            # 429 despite pacing (shared account, other processes): back off.
            await asyncio.sleep(backoff)
            backoff *= 2
        return resp

    async def _get_data(self, path: str, what: str) -> Any:
        resp = await self._request("GET", path)
        if resp.status_code >= 300:
            raise BellhopError(f"{what} failed ({resp.status_code}): {self._err_text(resp)}")
        return resp.json()["data"]

    async def instance_types(self) -> dict[str, Any]:
        """The live catalog: type name -> {instance_type, regions_with_capacity_available}."""
        return await self._get_data("/instance-types", "instance_types")

    async def launch(self, body: dict[str, Any]) -> str:
        resp = await self._request("POST", "/instance-operations/launch", json=body)
        if resp.status_code >= 300:
            raise ProvisionError(f"launch failed ({resp.status_code}): {self._err_text(resp)}")
        ids = resp.json()["data"]["instance_ids"]
        if not ids:
            raise ProvisionError(f"launch returned no instance ids: {resp.text[:300]}")
        return ids[0]

    async def get_instance(self, instance_id: str) -> dict[str, Any]:
        return await self._get_data(f"/instances/{instance_id}", "get_instance")

    async def list_instances(self) -> list[dict[str, Any]]:
        return await self._get_data("/instances", "list_instances")

    async def terminate(self, instance_ids: list[str]) -> None:
        resp = await self._request("POST", "/instance-operations/terminate",
                                   json={"instance_ids": instance_ids})
        # 404 = already gone; treat as success (idempotent teardown).
        if resp.status_code >= 300 and resp.status_code != 404:
            raise BellhopError(f"terminate failed ({resp.status_code}): {self._err_text(resp)}")

    async def ssh_keys(self) -> list[dict[str, Any]]:
        return await self._get_data("/ssh-keys", "ssh_keys")

    async def add_ssh_key(self, name: str, public_key: str) -> dict[str, Any]:
        resp = await self._request("POST", "/ssh-keys",
                                   json={"name": name, "public_key": public_key})
        if resp.status_code >= 300:
            raise BellhopError(f"add_ssh_key failed ({resp.status_code}): {self._err_text(resp)}")
        return resp.json()["data"]


@dataclass
class LambdaConfig:
    """Config for a Lambda Cloud instance. The Lambda-side peer of ``PodConfig``.

    Lambda is GPU-only, so there is no CPU spelling: set ``gpu=`` (canonical
    short name, or a verbatim ``gpu_...`` type name) or ``instance_type=``.
    """

    gpu: str | None = None                 # canonical short name ("H100", …); "gpu_..." passes verbatim
    instance_type: str | None = None       # verbatim Lambda type name (peer of PodConfig.gpu_id)
    gpu_count: int = 1                     # crossed with gpu= aliases (ignored for verbatim names)
    region: str | None = None              # e.g. "us-east-1"; None = any region with live capacity
    image: str | dict | None = None        # image family (str) or {"id": ...}; None = latest Lambda Stack
    file_system_names: list[str] = field(default_factory=list)   # persistent FS to attach
    user_data: str | None = None           # cloud-init passthrough (≤1MB)
    # pip specs installed right after readiness, before the instance is
    # yielded — same semantics as PodConfig.pip.
    pip: list[str] = field(default_factory=list)
    name: str = "bellhop"                  # launch stamps a -t<epoch> suffix on (see gc_instances)
    # auth / connection
    ssh_key: str | None = None             # private key; default ~/.ssh/id_ed25519
    ssh_user: str = "ubuntu"               # fixed by Lambda's images
    ssh_key_name: str | None = None        # pre-registered Lambda key name; default: auto-register local pubkey
    # readiness. Defaults resolve in __post_init__: Lambda boots slower than
    # RunPod (~3-5 min for 1x types, 10-15 min for 8x), so the provision
    # window scales with gpu_count. The waits exit early on success.
    ready: ReadyProbe = field(default_factory=lambda: SshProbe("true"))
    provision_timeout: timedelta | None = None
    ready_timeout: timedelta | None = None
    poll_interval: float = 10.0            # stays inside the 1 req/s budget
    # CLIENT-SIDE watchdog only — Lambda has no server-side TTL at all, so
    # this dies with your process. `bellhop lambda gc` is the real backstop.
    max_lifetime: timedelta | None = None

    def __post_init__(self):
        big = self.gpu_count >= 8 or "8x" in (self.instance_type or self.gpu or "")
        if self.provision_timeout is None:
            self.provision_timeout = timedelta(seconds=1800 if big else 900)
        if self.ready_timeout is None:
            self.ready_timeout = timedelta(seconds=600)

    def resolve_instance_types(self) -> list[str]:
        """The Lambda type names this config asks for, in preference order."""
        if self.gpu and self.instance_type:
            raise PreflightError("set gpu= (canonical name) or instance_type= (verbatim Lambda name), not both")
        if self.instance_type:
            return [self.instance_type]
        if not self.gpu:
            raise PreflightError("Lambda Cloud is GPU-only; set gpu= (e.g. gpu='H100') or instance_type=")
        if self.gpu.startswith("gpu_"):
            return [self.gpu]  # full Lambda type name, pass verbatim (count is baked in)
        hit = _ALIAS_LOOKUP.get(_canon_gpu(self.gpu))
        if hit:
            return [f"gpu_{self.gpu_count}x_{v}" for v in hit]
        raise PreflightError(
            f"unknown gpu {self.gpu!r}; known aliases: {sorted(LAMBDA_GPU_ALIASES)} "
            "(a full Lambda type name like 'gpu_8x_h100_sxm5' also works)"
        )

    def resolve_ssh_key(self) -> str:
        return resolve_ssh_key(self.ssh_key)

    def pubkey_text(self) -> str:
        return pubkey_text(self.ssh_key)

    def to_launch_body(self, instance_type: str, region: str, ssh_key_name: str) -> dict:
        body: dict = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key_name],  # API: exactly one, pre-registered
            # The -t<epoch> stamp is gc_instances's only age signal — the
            # Instance record has no created-at field.
            "name": f"{self.name[:48]}-t{int(time.time())}",
        }
        if self.image:
            body["image"] = {"family": self.image} if isinstance(self.image, str) else dict(self.image)
        if self.file_system_names:
            body["file_system_names"] = list(self.file_system_names)
        if self.user_data:
            body["user_data"] = self.user_data
        return body


class LambdaInstance(SshBox):
    """A live Lambda instance. Construct via :func:`instance` (the async CM).

    Transport comes from :class:`SshBox`; unlike a RunPod pod there is no NAT —
    the box is ``ubuntu@<public ip>:22`` directly, so ``mapped_port`` is the
    identity and the standard probes work unchanged.
    """

    _noun = "instance"

    def __init__(self, rest: LambdaRest, instance_id: str, config: LambdaConfig):
        self._rest = rest
        self.id = instance_id
        self.config = config
        self._meta: dict = {}
        self._ssh_key = config.resolve_ssh_key()
        self._lifetime_expired = False

    # ---- connection info ---------------------------------------------------
    @property
    def ssh_user(self) -> str:
        return self.config.ssh_user

    @property
    def host(self) -> str | None:
        return self._meta.get("ip")

    def mapped_port(self, container_port: int = 22) -> int:
        return container_port  # direct networking, no NAT mapping

    @property
    def status(self) -> str:
        return (self._meta.get("status") or "unknown").lower()

    # ---- lifecycle ---------------------------------------------------------
    async def refresh(self) -> dict:
        self._meta = await self._rest.get_instance(self.id)
        return self._meta

    async def _wait_provision(self) -> None:
        deadline = time.monotonic() + self.config.provision_timeout.total_seconds()
        while True:
            await self.refresh()
            if self.status in ("terminated", "terminating", "preempted"):
                raise ProvisionError(f"instance {self.id} entered terminal state {self.status}")
            # "unhealthy" is left to the timeout: it can be a transient boot
            # phase (SkyPilot treats it as still-initializing), and failing
            # fast would abandon a box that was about to come good.
            if self.status == "active" and self.host:
                return
            if time.monotonic() >= deadline:
                raise PodNotReadyError(
                    f"instance {self.id} not active+addressed within "
                    f"{self.config.provision_timeout.total_seconds():.0f}s (status={self.status})"
                )
            await asyncio.sleep(self.config.poll_interval)

    async def _wait_ready(self) -> None:
        await wait_ready(self, self.config.ready, self.config.ready_timeout,
                         self.config.poll_interval)

    async def teardown(self) -> None:
        await self._rest.terminate([self.id])

    # ---- exec / transfer: inherited from SshBox ------------------------------
    def _ssh_endpoint(self) -> tuple[str, int]:
        if not self.host:
            raise PodNotReadyError("instance has no public ip yet")
        return self.host, 22


def _pubkey_blob(text: str) -> str:
    """type+base64 of an OpenSSH public key — the comparable part (comment varies)."""
    parts = text.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else text.strip()


async def _ensure_ssh_key(rest: LambdaRest, config: LambdaConfig) -> str:
    """The registered key name to launch with, registering the local key if needed."""
    if config.ssh_key_name:
        names = [k["name"] for k in await rest.ssh_keys()]
        if config.ssh_key_name not in names:
            raise PreflightError(
                f"ssh_key_name {config.ssh_key_name!r} is not registered with Lambda "
                f"(have {names}); leave it unset to auto-register your local key"
            )
        return config.ssh_key_name
    pub = config.pubkey_text()
    blob = _pubkey_blob(pub)
    keys = await rest.ssh_keys()
    for k in keys:
        if _pubkey_blob(k.get("public_key", "")) == blob:
            return k["name"]
    name = f"bellhop-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"
    try:
        await rest.add_ssh_key(name, pub)
    except BellhopError:
        # Lost a registration race (or a stale same-name key): re-list and
        # match by blob; only fail if the key genuinely isn't there.
        for k in await rest.ssh_keys():
            if _pubkey_blob(k.get("public_key", "")) == blob:
                return k["name"]
        raise
    return name


async def _launch(rest: LambdaRest, config: LambdaConfig, ssh_key_name: str) -> str:
    """Walk (type, region) candidates against the live catalog; return an instance id."""
    catalog = await rest.instance_types()
    wanted = config.resolve_instance_types()
    known = [t for t in wanted if t in catalog]
    if not known:
        raise PreflightError(
            f"no Lambda instance type matches {wanted} "
            f"(gpu={config.gpu!r} x{config.gpu_count}); catalog has: {sorted(catalog)}"
        )
    attempts: list[tuple[str, str]] = []
    errors: list[str] = []
    for t in known:
        live = [r["name"] for r in catalog[t].get("regions_with_capacity_available") or []]
        if config.region:
            # The capacity list is advisory and launch is the source of truth:
            # attempt a pinned region even when the catalog says it's dry.
            attempts.append((t, config.region))
        elif live:
            attempts.extend((t, r) for r in live)
        else:
            errors.append(f"{t}: no live capacity in any region")
    if not attempts:
        raise ProvisionError(
            "lambda launch: no capacity for any candidate:\n  " + "\n  ".join(errors)
        )
    for t, region in attempts[:_MAX_LAUNCH_ATTEMPTS]:
        try:
            return await rest.launch(config.to_launch_body(t, region, ssh_key_name))
        except ProvisionError as e:
            errors.append(f"{t} in {region}: {e}")
    raise ProvisionError(
        "launch failed on every type/region attempt:\n  " + "\n  ".join(errors)
    )


@contextlib.asynccontextmanager
async def instance(config: LambdaConfig, *, keep: bool = False,
                   api_key: str | None = None) -> AsyncIterator[LambdaInstance]:
    """Launch a Lambda instance, wait until it's functional, yield it, terminate it.

    On any exception (including a readiness timeout) the instance is still
    terminated, unless ``keep=True``. There is NO server-side TTL on Lambda:
    ``keep=True`` (or a killed host) leaves the box billing until you
    terminate it — ``bellhop lambda gc`` reaps stamped leftovers.
    """
    config.resolve_instance_types()          # preflight before spending money
    config.resolve_ssh_key()
    if config.max_lifetime:
        warnings.warn(
            "Lambda has no server-side TTL: max_lifetime is enforced by an "
            "in-process watchdog only — if this process dies, nothing terminates "
            "the instance (sweep leaks with `bellhop lambda gc`)",
            stacklevel=3,
        )
    async with LambdaRest(api_key=api_key) as rest:
        ssh_key_name = await _ensure_ssh_key(rest, config)
        instance_id = await _launch(rest, config, ssh_key_name)
        inst = LambdaInstance(rest, instance_id, config)
        watchdog: asyncio.Task | None = None
        try:
            await inst._wait_provision()
            await inst._wait_ready()
            await ensure_workspace(inst)
            if config.pip:
                await install_pip(inst, config.pip)
            if config.max_lifetime:
                watchdog = asyncio.create_task(lifetime_watchdog(inst, config.max_lifetime))
            try:
                yield inst
            except BaseException as e:
                if inst._lifetime_expired:
                    # The watchdog killed the box mid-run; say so instead of
                    # letting the ssh exit-255 masquerade as a job failure.
                    raise BellhopError(
                        f"instance {inst.id} hit max_lifetime "
                        f"{config.max_lifetime} mid-run and was terminated"
                    ) from e
                raise
        finally:
            if watchdog:
                watchdog.cancel()
            if keep and config.max_lifetime and not inst._lifetime_expired:
                warnings.warn(
                    f"keep=True disarms the max_lifetime watchdog: instance {inst.id} "
                    "now has NO lifetime bound (reap with `bellhop lambda gc`)",
                    stacklevel=3,
                )
            if not keep:
                with contextlib.suppress(Exception):
                    await inst.teardown()


# ---- leak reaping (no server-side TTL => leaks are on us to find) -----------

def _stamp_epoch(name: str | None) -> int | None:
    m = _NAME_STAMP.match(name or "")
    return int(m.group(1)) if m else None


async def list_instances(api_key: str | None = None) -> list[dict]:
    """All of the account's instances (bellhop-launched or not)."""
    async with LambdaRest(api_key=api_key) as rest:
        return await rest.list_instances()


async def gc_instances(older_than: timedelta, *, dry_run: bool = False,
                       api_key: str | None = None) -> list[dict]:
    """Terminate bellhop-stamped instances older than ``older_than``.

    Only instances whose *name* carries the ``bellhop…-t<epoch>`` stamp are
    candidates — other instances (yours or a colleague's) are never touched,
    and a bellhop-named instance without a parseable stamp is left alone too.
    Returns the reaped (or would-reap) instances with an ``age_hours`` field.
    """
    now = time.time()
    reaped: list[dict] = []
    async with LambdaRest(api_key=api_key) as rest:
        for inst in await rest.list_instances():
            epoch = _stamp_epoch(inst.get("name"))
            if epoch is None or inst.get("status") in ("terminated", "terminating"):
                continue
            age = now - epoch
            if age < older_than.total_seconds():
                continue
            reaped.append({**inst, "age_hours": round(age / 3600, 1)})
        if not dry_run and reaped:
            await rest.terminate([i["id"] for i in reaped])
    return reaped
