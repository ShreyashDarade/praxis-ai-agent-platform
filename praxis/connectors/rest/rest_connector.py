# praxis/connectors/rest/rest_connector.py
"""Generic JSON-over-HTTP connector - the universal escape hatch (spec §6).

Every other connector in this package knows something specific about
the system on the far side: `GitHubConnector` knows `/repos/{owner}/{repo}`,
`PrometheusConnector` knows PromQL, `ElasticsearchConnector` knows the
query DSL. This one knows nothing, on purpose. It is what a deployment
registers when it has to reach an HTTP API that has no dedicated
connector and is not worth writing one for - spec §6's "adding a new
connector type is ... a config entry - no core code change", taken to
its limit: for a plain JSON API, even the config entry is enough.

Built on httpx, like `GitHubConnector` and `PrometheusConnector`, for
the same reason those are: an HTTP API reached with a bearer token and
JSON needs no SDK.

**What this connector genuinely cannot do, and does not pretend to:**

- It has no schema. A REST API is not self-describing. `describe()`
  makes exactly one attempt at a machine-readable description - fetching
  an OpenAPI document from a conventional path - and when that is absent
  it says so explicitly in its returned schema rather than returning an
  empty dict that a caller (or the Capability Factory, which JSON-dumps
  `describe()` output straight into a synthesis prompt) could read as
  "this API has no endpoints".
- It speaks JSON only. A response that is not JSON is a
  `ConnectorQueryError`, not a silently stringified body - a caller
  asking this connector for data has to be able to trust that what comes
  back is parsed data.
- Its `health()` can only tell reachability from health. See that
  method's own docstring: a generic HTTP base URL has no agreed health
  endpoint, and a 404 at the root of an otherwise perfectly working API
  is completely normal.
- It does no SSRF validation, no robots.txt check, and no byte cap.
  Those belong to `praxis.connectors.web.WebConnector`, which exists
  precisely because reaching *the open web* needs all of that. This
  connector is for a URL an operator deliberately configured, the same
  trust posture as a configured database DSN - do not point it at a
  user-supplied or LLM-generated host.
"""
from __future__ import annotations

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

_DEFAULT_TIMEOUT_SECONDS = 15.0
_DEFAULT_AUTH_HEADER_NAME = "Authorization"
_DEFAULT_OPENAPI_PATH = "/openapi.json"

# How much of a non-JSON body to quote back in the resulting error. Long
# enough to recognise an HTML error page or a proxy's plain-text notice,
# short enough that a megabyte of HTML never lands in a log line or an
# LLM's context window.
_BODY_SNIPPET_CHARS = 200

_WRITE_METHODS = {"post": "POST", "put": "PUT", "patch": "PATCH", "delete": "DELETE"}

# An OpenAPI Path Item Object mixes operations in with keys that are not
# operations at all - `parameters`, `summary`, `description`, `servers`,
# `$ref`. Whitelisting the eight method names the spec defines is the
# only correct way to separate them; excluding the known non-methods
# instead would silently report a future sibling key as an HTTP verb.
_OPENAPI_METHODS = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)


class RESTConnector(Connector):
    """One configured JSON-over-HTTP API.

    `base_url` is joined with the path passed to `read()`/`write()` by
    httpx's own base-URL merge, which means a path must be relative to
    whatever prefix `base_url` already carries. Note httpx's rule here,
    because it surprises people: a *leading slash* on the request path
    does not discard `base_url`'s own path the way `urljoin` would -
    httpx concatenates them, so `base_url="https://api.example.com/v2"`
    plus `"/users"` correctly requests `/v2/users`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        name: str = "rest",
        read_only: bool = True,
        auth_header_name: str | None = None,
        auth_header_value: str | None = None,
        openapi_path: str = _DEFAULT_OPENAPI_PATH,
        health_path: str = "/",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self._base_url = base_url.rstrip("/")
        self._auth_header_name = auth_header_name or _DEFAULT_AUTH_HEADER_NAME
        self._auth_header_value = auth_header_value
        self._openapi_path = openapi_path
        self._health_path = health_path
        self._timeout_seconds = timeout_seconds

    def _client(self) -> httpx.AsyncClient:
        headers = {"Accept": "application/json"}
        if self._auth_header_value:
            headers[self._auth_header_name] = self._auth_header_value
        return httpx.AsyncClient(
            base_url=self._base_url, headers=headers, timeout=self._timeout_seconds
        )

    async def describe(self) -> ConnectorDescription:
        """Best-effort OpenAPI discovery, with an explicit answer either way.

        A REST API has no introspection endpoint the way GraphQL and
        Elasticsearch do. The one widely-honoured convention is an
        OpenAPI document at a well-known path, so that is what this
        tries - and `"openapi_available": False` plus a `"note"`
        explaining why is what comes back when there isn't one. That
        negative answer is the point: the alternative, an empty
        `{"endpoints": {}}`, is indistinguishable from an API that
        genuinely exposes nothing.

        A 404/406/non-JSON response at the OpenAPI path is the normal
        case, not a failure, and is reported rather than raised. A
        *transport* failure still raises `ConnectorUnavailableError` -
        "the host is unreachable" is not the same fact as "this API
        publishes no OpenAPI document", and collapsing the two would
        make an entirely dead endpoint look merely undocumented.
        """
        try:
            async with self._client() as client:
                response = await client.get(self._openapi_path)
        except httpx.HTTPError as exc:
            raise ConnectorUnavailableError(
                f"connector '{self.name}' could not reach '{self._base_url}'",
                connector_name=self.name,
                detail=describe_exception(exc),
            ) from exc

        schema: dict[str, Any] = {
            "base_url": self._base_url,
            "authenticated": self._auth_header_value is not None,
            "read_only": self.read_only,
        }

        document = _json_or_none(response)
        if response.status_code != 200 or not isinstance(document, dict) or "paths" not in document:
            schema["openapi_available"] = False
            schema["note"] = (
                f"no OpenAPI document at '{self._openapi_path}' (HTTP "
                f"{response.status_code}); this API's endpoints are not "
                "machine-discoverable and must be supplied by the caller"
            )
            return ConnectorDescription(kind="rest", schema=schema)

        paths = document.get("paths")
        # Bound once and narrowed, rather than `.get("info")` twice:
        # the second call is a fresh lookup the type checker cannot
        # tie to the first one's isinstance check.
        raw_info = document.get("info")
        info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
        schema["openapi_available"] = True
        schema["title"] = info.get("title")
        schema["version"] = info.get("version")
        # Reduced to path -> methods rather than passed through whole: a
        # real OpenAPI document is frequently hundreds of kilobytes of
        # per-parameter schemas, and `CapabilityFactory` JSON-dumps this
        # dict verbatim into a synthesis prompt.
        schema["endpoints"] = {
            path: sorted(
                method.upper() for method in operations if method.lower() in _OPENAPI_METHODS
            )
            for path, operations in (paths.items() if isinstance(paths, dict) else [])
            if isinstance(operations, dict)
        }
        return ConnectorDescription(kind="rest", schema=schema)

    async def read(self, query: str, **params: Any) -> Any:
        """GETs `query` (a path relative to `base_url`) and returns parsed JSON.

        Every keyword argument becomes a query-string parameter, the
        same convention `PrometheusConnector.read()` already uses - so
        ``read("/users", page=2, per_page=50)`` requests
        ``/users?page=2&per_page=50``. That means this connector has no
        reserved kwargs of its own: anything passed goes on the wire.
        """
        try:
            async with self._client() as client:
                response = await client.get(query, params=params or None)
        except httpx.HTTPError as exc:
            raise ConnectorUnavailableError(
                f"connector '{self.name}' could not GET '{query}'",
                connector_name=self.name,
                detail=describe_exception(exc),
            ) from exc
        return self._parse(response, f"GET {query}")

    async def write(self, action: str, **params: Any) -> Any:
        """`action` is an HTTP method name: "post", "put", "patch" or "delete".

        `params` must carry `path`; an optional `json` becomes the
        request body and an optional `params` dict becomes the query
        string. Unlike `read()`, stray keyword arguments are *not*
        swept into the query string - a typo in a write's argument name
        silently becoming a query parameter is a much worse failure than
        it is on a read, so anything unrecognised is refused.
        """
        if self.read_only:
            raise ConnectorReadOnlyError(
                f"connector '{self.name}' is registered read-only; refusing action '{action}'",
                connector_name=self.name,
            )

        method = _WRITE_METHODS.get(action.lower())
        if method is None:
            raise NotImplementedError(
                f"unsupported action '{action}' for connector '{self.name}' "
                f"(expected one of {sorted(_WRITE_METHODS)})"
            )

        path = params.pop("path", None)
        if not isinstance(path, str) or not path:
            raise ConnectorQueryError(
                f"connector '{self.name}' action '{action}' requires a non-empty 'path'",
                connector_name=self.name,
            )
        body = params.pop("json", None)
        query_params = params.pop("params", None)
        if params:
            raise ConnectorQueryError(
                f"connector '{self.name}' action '{action}' got unexpected arguments",
                connector_name=self.name,
                detail=f"unrecognised: {sorted(params)}; expected path/json/params",
            )

        try:
            async with self._client() as client:
                response = await client.request(method, path, json=body, params=query_params)
        except httpx.HTTPError as exc:
            raise ConnectorUnavailableError(
                f"connector '{self.name}' could not {method} '{path}'",
                connector_name=self.name,
                detail=describe_exception(exc),
            ) from exc

        # A write that legitimately returns no body (204 No Content, and
        # in practice plenty of 200s from DELETE handlers) is a success,
        # not a parse failure - so this one case returns a synthetic
        # result rather than going through `_parse`.
        if response.status_code == 204 or not response.content:
            _raise_for_status(response, self.name, f"{method} {path}")
            return {"status_code": response.status_code}
        return self._parse(response, f"{method} {path}")

    async def health(self) -> HealthStatus:
        """Reports *reachability*, which is all a generic HTTP connector can honestly report.

        There is no cross-API health endpoint. `health_path` defaults to
        the base URL's own root, and plenty of perfectly healthy APIs
        answer 404 or 401 there. So the rule is deliberately coarse and
        stated rather than hidden: any HTTP response below 500 means the
        server answered and is therefore up (the status is recorded in
        `detail` when it isn't a 2xx, so an operator can see *what* it
        answered); a 5xx or a transport failure means it is not.

        The honest limitation: this cannot detect an API that is up but
        broken behind the root path, and it will report a
        misconfigured-but-answering proxy as healthy. A deployment that
        needs a real check should pass a `health_path` its API actually
        implements.
        """
        try:
            async with self._client() as client:
                response = await client.get(self._health_path)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))

        if response.status_code >= 500:
            return HealthStatus(
                name=self.name,
                healthy=False,
                detail=f"unexpected status {response.status_code}",
            )
        detail = "" if response.is_success else f"reachable, status {response.status_code}"
        return HealthStatus(name=self.name, healthy=True, detail=detail)

    def _parse(self, response: httpx.Response, operation: str) -> Any:
        _raise_for_status(response, self.name, operation)
        payload = _json_or_none(response)
        if payload is None:
            snippet = response.text[:_BODY_SNIPPET_CHARS]
            raise ConnectorQueryError(
                f"connector '{self.name}' expected a JSON body from {operation}",
                connector_name=self.name,
                detail=(
                    f"content-type '{response.headers.get('content-type', 'unset')}', "
                    f"body starts: {snippet!r}"
                ),
                status=response.status_code,
            )
        return payload


def _json_or_none(response: httpx.Response) -> Any:
    """`response.json()` or `None` if the body is not JSON at all.

    Returning `None` for "not JSON" is safe here (rather than ambiguous
    with a body that is literally ``null``) because every call site
    treats a bare ``null`` body from a data API as unusable anyway - and
    the alternative, a sentinel object, would buy nothing.
    """
    try:
        return response.json()
    except ValueError:
        return None


def _raise_for_status(response: httpx.Response, connector_name: str, operation: str) -> None:
    """Non-2xx -> `ConnectorQueryError` carrying the server's own body.

    Deliberately not `response.raise_for_status()`: httpx's own error
    message says only "Client error '404 Not Found' for url ...", and
    discards the response body - which for a JSON API is where the
    actual reason lives. The body is what a caller needs to fix the
    call, so it is quoted (bounded) into `detail`.
    """
    if response.is_success:
        return
    raise ConnectorQueryError(
        f"connector '{connector_name}' got HTTP {response.status_code} from {operation}",
        connector_name=connector_name,
        detail=response.text[:_BODY_SNIPPET_CHARS],
        status=response.status_code,
    )


register_connector_factory(
    ConnectorFactory(
        name="rest",
        is_configured=lambda settings: bool(settings.rest_base_url),
        build=lambda settings: RESTConnector(
            base_url=required(settings.rest_base_url, setting="rest_base_url"),
            auth_header_name=settings.rest_auth_header_name,
            auth_header_value=settings.rest_auth_header_value,
        ),
    )
)
