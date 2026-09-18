# praxis/connectors/graphql/graphql_connector.py
"""GraphQL connector (spec §6).

Built on httpx rather than a GraphQL client library, and that is a
considered choice rather than the lazy one. A GraphQL request over HTTP
is a POST of ``{"query": ..., "variables": ...}`` to a single endpoint -
there is no protocol here for a client library to abstract. What the
client libraries actually add is a schema-aware *document builder* and
codegen, both of which assume you know the schema at build time; this
connector exists to talk to schemas nobody knew about at build time, so
that value does not apply. `graphql-core` would have been worth adding
for one thing only - parsing the query document to classify its
operations - and `_top_level_operations` below does that job for this
one narrow purpose without a dependency whose main feature set
(execution, validation, a full server runtime) Praxis would never use.

Two things this connector refuses to do quietly, because both are
GraphQL-specific traps:

1. **A GraphQL error arrives with HTTP 200.** Spec-conformant servers
   return `{"errors": [...]}` under a 200, so an `httpx`
   `raise_for_status()`-shaped connector would hand a caller a payload
   full of nulls and call it a success. `read()` raises
   `ConnectorQueryError` on any non-empty `errors` array.
2. **A mutation is just a query document with a different keyword.**
   Nothing about the transport distinguishes a read from a write, so a
   read-only GraphQL connector that only guarded `write()` would guard
   nothing at all - `read()` would happily POST a mutation. See
   `_top_level_operations`.
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

_DEFAULT_TIMEOUT_SECONDS = 20.0
_DEFAULT_AUTH_HEADER_NAME = "Authorization"
_BODY_SNIPPET_CHARS = 200

_MUTATING_OPERATIONS = {"mutation", "subscription"}

# Deliberately not the full introspection query the GraphiQL IDE sends.
# That one pulls every argument, every input field, every enum value,
# every directive and the full deprecation metadata - hundreds of
# kilobytes on a real schema, and `describe()` output is JSON-dumped
# verbatim into a Capability Factory synthesis prompt. This asks for
# exactly what "what can I query here" needs: the root operation type
# names, and each type's own field names with their (unwrapped) type
# names.
_INTROSPECTION_QUERY = """
query PraxisIntrospection {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      name
      kind
      fields(includeDeprecated: false) {
        name
        type { name kind ofType { name kind ofType { name kind } } }
      }
    }
  }
}
""".strip()


class GraphQLErrorsReturned(ConnectorQueryError):
    """The server answered with a non-empty `errors` array.

    A distinct type, rather than a bare `ConnectorQueryError` with an
    attribute bolted on afterwards, because `partial_data` only ever
    exists for this one failure mode and a caller should be able to test
    for it rather than probe with `getattr`. GraphQL allows `errors`
    alongside a non-null `data` when one field of many resolved badly,
    and `partial_data` is whatever did resolve - `None` when the whole
    operation failed.
    """

    def __init__(
        self,
        message: str,
        *,
        connector_name: str,
        detail: str = "",
        status: int | str | None = None,
        partial_data: Any = None,
    ) -> None:
        super().__init__(message, connector_name=connector_name, detail=detail, status=status)
        self.partial_data = partial_data


class GraphQLConnector(Connector):
    """One configured GraphQL endpoint."""

    def __init__(
        self,
        url: str,
        *,
        name: str = "graphql",
        read_only: bool = True,
        auth_header_name: str | None = None,
        auth_header_value: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self._url = url
        self._auth_header_name = auth_header_name or _DEFAULT_AUTH_HEADER_NAME
        self._auth_header_value = auth_header_value
        self._timeout_seconds = timeout_seconds

    def _client(self) -> httpx.AsyncClient:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._auth_header_value:
            headers[self._auth_header_name] = self._auth_header_value
        return httpx.AsyncClient(headers=headers, timeout=self._timeout_seconds)

    async def describe(self) -> ConnectorDescription:
        """Runs an introspection query and reduces the schema to type -> field names.

        **The limitation that matters:** introspection is disabled on a
        great many production GraphQL endpoints (it is the default
        posture for Apollo Server in production, and a standard
        hardening step elsewhere). When it is, the server answers 200
        with an `errors` array saying so, and this raises
        `ConnectorQueryError` carrying that message - it does not return
        an empty schema, because "this endpoint exposes nothing" and
        "this endpoint will not tell me what it exposes" are completely
        different facts and only one of them is fixable by an operator.

        Introspection-internal types (the `__`-prefixed ones) and types
        with no fields at all (scalars, enums, unions) are dropped:
        neither tells a caller anything about what it can query.
        """
        payload = await self._post(
            {"query": _INTROSPECTION_QUERY, "variables": {}}, operation="introspection"
        )
        schema_root = (payload.get("data") or {}).get("__schema")
        if not isinstance(schema_root, dict):
            raise ConnectorQueryError(
                f"connector '{self.name}' got no __schema back from its introspection query",
                connector_name=self.name,
                detail=f"response data was {payload.get('data')!r}",
            )

        types: dict[str, list[dict[str, str]]] = {}
        for type_entry in schema_root.get("types") or []:
            if not isinstance(type_entry, dict):
                continue
            type_name = type_entry.get("name") or ""
            if type_name.startswith("__") or not type_entry.get("fields"):
                continue
            types[type_name] = [
                {"name": field["name"], "type": _unwrap_type_name(field.get("type"))}
                for field in type_entry["fields"]
                if isinstance(field, dict) and field.get("name")
            ]

        return ConnectorDescription(
            kind="graphql",
            schema={
                "url": self._url,
                "read_only": self.read_only,
                "query_type": (schema_root.get("queryType") or {}).get("name"),
                "mutation_type": (schema_root.get("mutationType") or {}).get("name"),
                "subscription_type": (schema_root.get("subscriptionType") or {}).get("name"),
                "types": types,
            },
        )

    async def read(self, query: str, **params: Any) -> Any:
        """POSTs `query` as a GraphQL document and returns its `data` object.

        `params["variables"]` is the variables map and
        `params["operation_name"]` selects one operation from a
        multi-operation document; nothing else is recognised, and unlike
        `RESTConnector.read()` stray kwargs are not swept anywhere -
        GraphQL has no query string for them to go to.

        When `read_only` is set (the default), a document containing a
        top-level `mutation` or `subscription` is refused before
        anything is sent. Subscriptions are refused for a second,
        independent reason: they are a long-lived stream, not a
        request/response, and this connector has no transport for one -
        accepting the POST would return a single frame and silently drop
        the rest of the subscription.
        """
        variables = params.pop("variables", None) or {}
        operation_name = params.pop("operation_name", None)
        if params:
            raise ConnectorQueryError(
                f"connector '{self.name}' read got unexpected arguments",
                connector_name=self.name,
                detail=f"unrecognised: {sorted(params)}; expected variables/operation_name",
            )

        operations = _top_level_operations(query)
        refused = operations & _MUTATING_OPERATIONS
        if refused and self.read_only:
            raise ConnectorReadOnlyError(
                f"connector '{self.name}' is registered read-only; refusing a document "
                f"whose top-level operation(s) include {sorted(refused)}",
                connector_name=self.name,
            )
        if "subscription" in operations:
            raise ConnectorQueryError(
                f"connector '{self.name}' cannot run a subscription over HTTP POST",
                connector_name=self.name,
                detail=(
                    "a subscription is a long-lived stream; this connector is "
                    "request/response only"
                ),
            )

        body: dict[str, Any] = {"query": query, "variables": variables}
        if operation_name:
            body["operationName"] = operation_name
        payload = await self._post(body, operation="query")
        return payload.get("data")

    async def write(self, action: str, **params: Any) -> Any:
        """`action` must be `"mutation"`; `params["query"]` is the mutation document.

        Accepts optional `variables` and `operation_name` exactly as
        `read()` does. The document is *not* re-checked for being a
        mutation - a caller that reaches `write()` on a writable
        connector has already declared intent, and a `query` document
        sent through here is merely a read taking the long way round.
        """
        if self.read_only:
            raise ConnectorReadOnlyError(
                f"connector '{self.name}' is registered read-only; refusing action '{action}'",
                connector_name=self.name,
            )
        if action != "mutation":
            raise NotImplementedError(
                f"unsupported action '{action}' for connector '{self.name}' (expected 'mutation')"
            )

        document = params.pop("query", None)
        if not isinstance(document, str) or not document.strip():
            raise ConnectorQueryError(
                f"connector '{self.name}' action 'mutation' requires a non-empty 'query' document",
                connector_name=self.name,
            )
        body: dict[str, Any] = {"query": document, "variables": params.pop("variables", None) or {}}
        operation_name = params.pop("operation_name", None)
        if operation_name:
            body["operationName"] = operation_name
        if params:
            raise ConnectorQueryError(
                f"connector '{self.name}' action 'mutation' got unexpected arguments",
                connector_name=self.name,
                detail=f"unrecognised: {sorted(params)}; expected query/variables/operation_name",
            )

        payload = await self._post(body, operation="mutation")
        return payload.get("data")

    async def health(self) -> HealthStatus:
        """POSTs `{ __typename }` - the cheapest document every GraphQL
        server can answer, with no schema knowledge and no introspection
        needed (`__typename` is a meta-field on every type, and unlike
        `__schema` it is not what introspection-disabling switches off).
        """
        try:
            async with self._client() as client:
                response = await client.post(self._url, json={"query": "{ __typename }"})
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
        if payload.get("errors"):
            return HealthStatus(
                name=self.name, healthy=False, detail=str(payload["errors"])[:_BODY_SNIPPET_CHARS]
            )
        return HealthStatus(name=self.name, healthy=True)

    async def _post(self, body: dict[str, Any], *, operation: str) -> dict[str, Any]:
        """One POST, with GraphQL's two-layer error model mapped onto the taxonomy.

        Layer one is HTTP (a 401 from a gateway, a 502 from a proxy);
        layer two is the `errors` array inside a 200. Both become a
        `ConnectorQueryError`, but a partial response - `errors`
        alongside a non-null `data`, which GraphQL genuinely allows when
        one field of many resolved badly - still raises, with the
        partial payload attached to the exception as `partial_data`.

        That is the deliberate call: returning partial data would make a
        half-failed query indistinguishable from a complete one at the
        call site, and a caller acting on silently-missing fields is a
        worse outcome than a caller handling an exception. Nothing is
        lost either way, because `partial_data` carries whatever did
        resolve for a caller that genuinely wants it.
        """
        try:
            async with self._client() as client:
                response = await client.post(self._url, json=body)
        except httpx.HTTPError as exc:
            raise ConnectorUnavailableError(
                f"connector '{self.name}' could not reach '{self._url}'",
                connector_name=self.name,
                detail=describe_exception(exc),
            ) from exc

        if not response.is_success:
            raise ConnectorQueryError(
                f"connector '{self.name}' got HTTP {response.status_code} for its {operation}",
                connector_name=self.name,
                detail=response.text[:_BODY_SNIPPET_CHARS],
                status=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ConnectorQueryError(
                f"connector '{self.name}' got a non-JSON body for its {operation}",
                connector_name=self.name,
                detail=f"body starts: {response.text[:_BODY_SNIPPET_CHARS]!r}",
                status=response.status_code,
            ) from exc

        if not isinstance(payload, dict):
            raise ConnectorQueryError(
                f"connector '{self.name}' got a non-object JSON body for its {operation}",
                connector_name=self.name,
                detail=f"got {type(payload).__name__}",
                status=response.status_code,
            )

        errors = payload.get("errors")
        if errors:
            raise GraphQLErrorsReturned(
                f"connector '{self.name}' {operation} returned GraphQL errors",
                connector_name=self.name,
                detail=str(errors)[:_BODY_SNIPPET_CHARS],
                status=response.status_code,
                partial_data=payload.get("data"),
            )
        return payload


def _unwrap_type_name(type_ref: Any) -> str:
    """Reduces a GraphQL type reference to the named type inside it.

    A field's type comes back as a nest of wrappers - `[User!]!` is
    `NON_NULL(LIST(NON_NULL(User)))` - where every wrapper has
    `name: null` and the real name sits at the bottom of the `ofType`
    chain. Nullability and list-ness are dropped here on purpose:
    `describe()` exists to tell a caller *what it can ask for*, and
    three extra levels of wrapper metadata per field would multiply the
    size of a synthesis prompt without changing which fields exist.
    """
    current = type_ref
    while isinstance(current, dict):
        if current.get("name"):
            return str(current["name"])
        current = current.get("ofType")
    return "unknown"


def _top_level_operations(document: str) -> set[str]:
    """The operation keywords appearing at the top level of `document`.

    Returns a subset of `{"query", "mutation", "subscription"}` -
    empty for an anonymous shorthand document (`{ user { id } }`),
    which GraphQL defines as a query.

    This is a scanner, not a parser, and the scope is exactly what the
    read-only guard needs: find the keywords that can only be operation
    definitions, which means finding the ones at nesting depth zero.
    Depth is tracked across `{}`, `()` and `[]` together, because a
    variable definition's default value can itself contain braces
    (`query Q($f: Filter = {kind: "mutation"})`) and a selection set can
    contain an argument named `mutation` - both would fool a plain
    substring search. String literals (single and block) and `#`
    comments are skipped for the same reason.

    What it deliberately does not do: validate the document, resolve
    fragments, or catch a *server-side* write triggered by something
    that is syntactically a query. No client-side check can catch the
    latter - a `query` field whose resolver writes is indistinguishable
    from one that does not - which is why a genuinely read-only
    deployment should also be using a credential the GraphQL server
    itself restricts to reads. This guard is the same class of control
    as `PostgresConnector`'s mutation regex: defense in depth, honestly
    bounded, not a proof.
    """
    depth = 0
    index = 0
    word: list[str] = []
    found: set[str] = set()
    length = len(document)

    def flush() -> None:
        if word and depth == 0:
            token = "".join(word)
            if token in ("query", "mutation", "subscription"):
                found.add(token)
        word.clear()

    while index < length:
        char = document[index]

        if char == "#":
            flush()
            newline = document.find("\n", index)
            index = length if newline == -1 else newline + 1
            continue

        if document.startswith('"""', index):
            flush()
            end = document.find('"""', index + 3)
            index = length if end == -1 else end + 3
            continue

        if char == '"':
            flush()
            index += 1
            while index < length:
                if document[index] == "\\":
                    index += 2
                    continue
                if document[index] == '"':
                    index += 1
                    break
                index += 1
            continue

        if char in "{([":
            flush()
            depth += 1
            index += 1
            continue

        if char in "})]":
            flush()
            depth -= 1
            index += 1
            continue

        if char.isalnum() or char == "_":
            word.append(char)
            index += 1
            continue

        flush()
        index += 1

    flush()
    return found


register_connector_factory(
    ConnectorFactory(
        name="graphql",
        is_configured=lambda settings: bool(settings.graphql_url),
        build=lambda settings: GraphQLConnector(
            url=required(settings.graphql_url, setting="graphql_url"),
            auth_header_name=settings.graphql_auth_header_name,
            auth_header_value=settings.graphql_auth_header_value,
        ),
    )
)
