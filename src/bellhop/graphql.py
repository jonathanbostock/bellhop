"""GraphQL client for the one thing REST v1 can't do: native pod TTL.

RunPod REST v1 has no ``terminateAfter``/``stopAfter`` field. The GraphQL
``podFindAndDeployOnDemand`` mutation does (this is the same path
``runpodctl --terminate-after`` uses). So when a TTL is requested, we create
the pod here; everything else (get / port discovery / delete) stays on REST.
"""

from __future__ import annotations

from typing import Any

import httpx

from .errors import ProvisionError
from .rest import _api_key

GRAPHQL_URL = "https://api.runpod.io/graphql"

_CREATE_POD = """
mutation createPod($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    imageName
    machineId
  }
}
"""


class RunpodGraphQL:
    """Minimal async GraphQL client.

    Auth is an ``Authorization: Bearer`` header (verified against the live
    API) — runpodctl's ``?api_key=`` query param would leak the key into URL
    logs on any intermediary.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {_api_key(api_key)}"},
        )

    async def __aenter__(self) -> "RunpodGraphQL":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post(
            GRAPHQL_URL, json={"query": query, "variables": variables},
        )
        if resp.status_code in (401, 403):
            # RunPod's unauthorized body is literally {"error":{}} — name the
            # actual problem instead of relaying an empty shrug.
            raise ProvisionError(
                f"graphql HTTP {resp.status_code} unauthorized: check RUNPOD_API_KEY "
                "(bellhop sends it as an Authorization: Bearer header to "
                f"{GRAPHQL_URL}; a key that works with runpodctl's ?api_key= "
                f"style but fails here is malformed/expired) — body: {resp.text[:200]}"
            )
        if resp.status_code >= 300:
            raise ProvisionError(f"graphql HTTP {resp.status_code}: {resp.text[:400]}")
        body = resp.json()
        if body.get("errors"):
            msgs = "; ".join(e.get("message", str(e)) for e in body["errors"])
            raise ProvisionError(f"graphql error: {msgs}")
        return body.get("data") or {}

    async def create_pod_on_demand(self, gql_input: dict[str, Any]) -> dict[str, Any]:
        data = await self._post(_CREATE_POD, {"input": gql_input})
        pod = data.get("podFindAndDeployOnDemand")
        if not pod:
            # null pod = no machine matched the request (e.g. out of stock)
            raise ProvisionError("podFindAndDeployOnDemand returned null (no capacity for the request)")
        return pod
