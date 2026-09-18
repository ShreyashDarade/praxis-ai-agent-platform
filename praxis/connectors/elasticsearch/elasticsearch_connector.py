# praxis/connectors/elasticsearch/elasticsearch_connector.py
"""Elasticsearch connector, over the REST API via plain httpx (spec §6).

**Why not the official `elasticsearch` client.** Three reasons, in
order of weight:

1. The client's major versions are *deliberately* coupled to server
   major versions (8.x refuses to talk to a 7.x cluster without an
   explicit compatibility switch, and vice versa). Praxis does not know
   which cluster an operator will point this at, so a pinned client
   would be a pinned *server requirement* smuggled into pyproject.toml.
2. The same wire protocol is spoken by OpenSearch, which forked at 7.10
   and which the official Elastic client now actively rejects (it
   product-checks the `X-elastic-product` response header and refuses
   to proceed). Talking REST directly is what makes this connector work
   against both, and against the many managed "Elasticsearch-compatible"
   services in between.
3. What is actually being used here is four endpoints - `_mapping`,
   `_search`, `_doc`, `_cluster/health` - all stable across every
   version in the field for a decade, all plain JSON over HTTP, which
   httpx (already a dependency, already the transport for three other
   connectors here) does completely.

The cost is real and worth stating: no automatic node discovery, no
connection pooling across hosts, no sniffing, no retry-on-another-node.
This connector talks to exactly the one URL it is configured with. A
deployment that needs multi-node failover should put a load balancer in
front of the cluster and point this at that, which is the normal
operational answer regardless of client library.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from praxis.connectors.errors import (
    ConnectorQueryError,
    ConnectorReadOnlyError,
    ConnectorUnavailableError,
    describe_exception,
)
from praxis.connectors.factory import (
    ConnectorFactory,
    register_connector_factory,
    required,
)
from praxis.core.interfaces import Connector, ConnectorDescription, HealthStatus

_DEFAULT_TIMEOUT_SECONDS = 30.0
_BODY_SNIPPET_CHARS = 300

# Injected when a search body names no `size`. Elasticsearch's own
# default is 10, which is almost always wrong for an analytical read and
# silently so - a caller that asked for "all orders over $1000" and got
# exactly 10 has no signal that it was truncated.
_DEFAULT_SIZE = 100

# Refused (not clamped) above this. Clamping would be the silent
# truncation this connector exists to avoid; the caller asked for a
# number and deserves to be told the number is not available rather than
# handed a smaller one that looks complete. Past ~10k Elasticsearch's own
# `index.max_result_window` refuses anyway, and the right answer becomes
# the scroll/PIT API, which this connector does not implement.
_MAX_SIZE = 1000


class ElasticsearchConnector(Connector):
    """One configured Elasticsearch/OpenSearch cluster endpoint.

    Authentication is either an API key (`api_key`, sent as
    ``Authorization: ApiKey <key>``) or HTTP Basic (`basic_auth`, a
    ``(username, password)`` pair). Only the API key is reachable from
    `Settings` - see `praxis.config.Settings.elasticsearch_api_key` for
    why - but `basic_auth` is a real constructor argument for a caller
    registering a connector programmatically against a cluster that
    only does basic auth.
    """

    def __init__(
        self,
        base_url: str,
        *,
        name: str = "elasticsearch",
        read_only: bool = True,
        api_key: str | None = None,
        basic_auth: tuple[str, str] | None = None,
        default_index: str = "_all",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._basic_auth = basic_auth
        self._default_index = default_index
        self._timeout_seconds = timeout_seconds

    def _client(self) -> httpx.AsyncClient:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"ApiKey {self._api_key}"
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            auth=self._basic_auth,
            timeout=self._timeout_seconds,
        )

    async def describe(self) -> ConnectorDescription:
        """Lists indices with their mapped field names and types.

        Reduced from the raw `_mapping` response, which nests every
        field under `mappings.properties` and carries per-field
        analyzer/format/index options that no caller of `describe()`
        needs. Object and `nested` fields are flattened to dotted paths
        (`author.name`), because that is the form a query DSL has to
        reference them by anyway.

        Two deliberate exclusions, both stated rather than silent:

        - Indices whose name starts with `.` are dropped. Those are
          Kibana saved objects, security indices, ILM state, monitoring
          - cluster internals, never the data a task is after, and on a
          real cluster there are dozens of them with thousands of
          fields between them.
        - Fields with no `type` (pure object containers) contribute
          their children but not themselves.

        A field that exists in documents but not in the mapping will not
        appear here at all. That is not a bug in this method: with
        dynamic mapping disabled, Elasticsearch genuinely does not index
        such a field, and it is not queryable.
        """
        response = await self._request("GET", "/_mapping", operation="describe")
        payload = _parse_json(response, self.name, "describe")
        if not isinstance(payload, dict):
            # `_mapping` always answers with an object keyed by index
            # name; anything else means something between here and the
            # cluster (a proxy, a captive portal, the wrong URL) answered
            # instead, and that must surface as a typed failure rather
            # than an AttributeError from the loop below.
            raise ConnectorQueryError(
                f"connector '{self.name}' got an unexpected _mapping payload",
                connector_name=self.name,
                detail=f"expected a JSON object keyed by index name, got {type(payload).__name__}",
                status=response.status_code,
            )

        indices: dict[str, dict[str, str]] = {}
        for index_name, index_body in payload.items():
            if index_name.startswith("."):
                continue
            mappings = (index_body or {}).get("mappings") or {}
            indices[index_name] = _flatten_properties(mappings.get("properties") or {})

        return ConnectorDescription(
            kind="elasticsearch",
            schema={
                "url": self._base_url,
                "read_only": self.read_only,
                "default_index": self._default_index,
                "indices": indices,
            },
        )

    async def read(self, query: str, **params: Any) -> Any:
        """Runs `query` - a JSON query-DSL *search body* string - and returns
        the parsed `_search` response.

        `query` is the body, not a whole request: ``{"query": {"match":
        {"title": "outage"}}, "sort": [...]}``. The index comes from
        `params["index"]`, falling back to this connector's
        `default_index` (`"_all"` unless configured otherwise) - it is a
        kwarg rather than a key inside the body because the index is
        part of the URL in Elasticsearch's own API, and putting it in
        the body would invent a Praxis-specific envelope that a caller
        who knows Elasticsearch would not expect.

        The full `_search` response is returned, not just the hits.
        `hits.total`, `aggregations` and `timed_out` are all things a
        caller legitimately needs, and `timed_out: true` in particular
        is Elasticsearch reporting a *partial* result - discarding the
        envelope would hide exactly the flag that says the hits are
        incomplete.
        """
        index = params.pop("index", None) or self._default_index
        if params:
            raise ConnectorQueryError(
                f"connector '{self.name}' read got unexpected arguments",
                connector_name=self.name,
                detail=f"unrecognised: {sorted(params)}; expected 'index'",
            )

        body = _parse_query_body(query, self.name)
        size = body.get("size")
        if size is None:
            body["size"] = _DEFAULT_SIZE
        elif not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ConnectorQueryError(
                f"connector '{self.name}' read got a non-integer 'size'",
                connector_name=self.name,
                detail=f"got {size!r}",
            )
        elif size > _MAX_SIZE:
            raise ConnectorQueryError(
                f"connector '{self.name}' refuses a search with size={size}",
                connector_name=self.name,
                detail=(
                    f"the per-query bound is {_MAX_SIZE}; narrow the query, or use "
                    "aggregations, rather than paging a large result set through this connector"
                ),
            )

        response = await self._request(
            "POST", f"/{index}/_search", json_body=body, operation=f"search on '{index}'"
        )
        return _parse_json(response, self.name, f"search on '{index}'")

    async def write(self, action: str, **params: Any) -> Any:
        """`action` is `"index_document"` or `"delete_document"`.

        `index_document` requires `index` and `document`, and takes an
        optional `doc_id` (Elasticsearch generates one when omitted -
        which means an omitted `doc_id` always *creates*, never
        updates). `delete_document` requires `index` and `doc_id`.

        Neither refreshes the index, so a document written here is not
        guaranteed visible to a `read()` issued immediately afterwards -
        Elasticsearch is near-real-time by design (a ~1s refresh
        interval), and forcing `refresh=true` on every write would be a
        significant, invisible performance cost on a real cluster. A
        caller that needs read-your-write must wait or refresh itself.
        """
        if self.read_only:
            raise ConnectorReadOnlyError(
                f"connector '{self.name}' is registered read-only; refusing action '{action}'",
                connector_name=self.name,
            )

        index = params.pop("index", None)
        if not isinstance(index, str) or not index:
            raise ConnectorQueryError(
                f"connector '{self.name}' action '{action}' requires a non-empty 'index'",
                connector_name=self.name,
            )
        doc_id = params.pop("doc_id", None)

        if action == "index_document":
            document = params.pop("document", None)
            if not isinstance(document, dict):
                raise ConnectorQueryError(
                    f"connector '{self.name}' action 'index_document' requires a 'document' object",
                    connector_name=self.name,
                    detail=f"got {type(document).__name__}",
                )
            _reject_extra(params, self.name, action, "index/doc_id/document")
            path = f"/{index}/_doc/{doc_id}" if doc_id else f"/{index}/_doc"
            response = await self._request(
                "PUT" if doc_id else "POST",
                path,
                json_body=document,
                operation=f"index into '{index}'",
            )
            return _parse_json(response, self.name, f"index into '{index}'")

        if action == "delete_document":
            if not isinstance(doc_id, str) or not doc_id:
                raise ConnectorQueryError(
                    f"connector '{self.name}' action 'delete_document' requires a 'doc_id'",
                    connector_name=self.name,
                )
            _reject_extra(params, self.name, action, "index/doc_id")
            response = await self._request(
                "DELETE", f"/{index}/_doc/{doc_id}", operation=f"delete from '{index}'"
            )
            return _parse_json(response, self.name, f"delete from '{index}'")

        raise NotImplementedError(
            f"unsupported action '{action}' for connector '{self.name}' "
            "(expected 'index_document' or 'delete_document')"
        )

    async def health(self) -> HealthStatus:
        """`GET /_cluster/health`, mapping the cluster's own colour.

        `green` and `yellow` are both healthy: yellow means every
        primary shard is assigned but some replica is not, which is the
        *normal, permanent* state of a single-node cluster (a replica
        cannot be placed on the same node as its primary) and of any
        cluster mid-rebalance. Reporting every single-node development
        cluster as unhealthy would make this check noise. `red` - at
        least one primary shard unassigned, so some data genuinely
        cannot be read - is unhealthy, and the colour is always put in
        `detail` so a yellow cluster is visible rather than merely
        "healthy".
        """
        try:
            async with self._client() as client:
                response = await client.get("/_cluster/health")
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))

        if not response.is_success:
            return HealthStatus(
                name=self.name,
                healthy=False,
                detail=f"unexpected status {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError:
            return HealthStatus(
                name=self.name,
                healthy=False,
                detail=f"non-JSON body: {response.text[:_BODY_SNIPPET_CHARS]!r}",
            )
        status = payload.get("status")
        return HealthStatus(
            name=self.name,
            healthy=status in ("green", "yellow"),
            detail=f"cluster status {status}",
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        operation: str,
    ) -> httpx.Response:
        try:
            async with self._client() as client:
                response = await client.request(method, path, json=json_body)
        except httpx.HTTPError as exc:
            raise ConnectorUnavailableError(
                f"connector '{self.name}' could not reach '{self._base_url}' for {operation}",
                connector_name=self.name,
                detail=describe_exception(exc),
            ) from exc

        if not response.is_success:
            raise ConnectorQueryError(
                f"connector '{self.name}' got HTTP {response.status_code} for {operation}",
                connector_name=self.name,
                # Elasticsearch puts the real cause - `index_not_found_exception`,
                # `parsing_exception` with the offending line and column - in
                # the body, so quoting it is the difference between an
                # actionable error and "404".
                detail=_error_reason(response),
                status=response.status_code,
            )
        return response


def _error_reason(response: httpx.Response) -> str:
    """Elasticsearch's `{"error": {"type": ..., "reason": ...}}`, reduced to a line."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:_BODY_SNIPPET_CHARS]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return f"{error.get('type')}: {error.get('reason')}"
    if isinstance(error, str):
        return error
    return json.dumps(payload)[:_BODY_SNIPPET_CHARS]


def _parse_json(response: httpx.Response, connector_name: str, operation: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ConnectorQueryError(
            f"connector '{connector_name}' got a non-JSON body for {operation}",
            connector_name=connector_name,
            detail=f"body starts: {response.text[:_BODY_SNIPPET_CHARS]!r}",
            status=response.status_code,
        ) from exc


def _parse_query_body(query: str, connector_name: str) -> dict[str, Any]:
    try:
        body = json.loads(query)
    except (TypeError, ValueError) as exc:
        raise ConnectorQueryError(
            f"connector '{connector_name}' read expects a JSON query-DSL body string",
            connector_name=connector_name,
            detail=describe_exception(exc),
        ) from exc
    if not isinstance(body, dict):
        raise ConnectorQueryError(
            f"connector '{connector_name}' read expects a JSON object as the search body",
            connector_name=connector_name,
            detail=f"got {type(body).__name__}",
        )
    return body


def _flatten_properties(properties: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """`mappings.properties` -> `{dotted.field.path: type}`."""
    flattened: dict[str, str] = {}
    for field_name, definition in properties.items():
        if not isinstance(definition, dict):
            continue
        path = f"{prefix}{field_name}"
        field_type = definition.get("type")
        if field_type:
            flattened[path] = str(field_type)
        nested = definition.get("properties")
        if isinstance(nested, dict):
            flattened.update(_flatten_properties(nested, prefix=f"{path}."))
    return flattened


def _reject_extra(params: dict[str, Any], connector_name: str, action: str, expected: str) -> None:
    if params:
        raise ConnectorQueryError(
            f"connector '{connector_name}' action '{action}' got unexpected arguments",
            connector_name=connector_name,
            detail=f"unrecognised: {sorted(params)}; expected {expected}",
        )


register_connector_factory(
    ConnectorFactory(
        name="elasticsearch",
        is_configured=lambda settings: bool(settings.elasticsearch_url),
        build=lambda settings: ElasticsearchConnector(
            base_url=required(settings.elasticsearch_url, setting="elasticsearch_url"),
            api_key=settings.elasticsearch_api_key,
        ),
    )
)
