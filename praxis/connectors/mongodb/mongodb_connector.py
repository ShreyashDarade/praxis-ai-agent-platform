# praxis/connectors/mongodb/mongodb_connector.py
"""MongoDB connector, built on motor (spec §6).

`motor` is MongoDB's own asyncio driver - a thin async facade over
`pymongo`, maintained by MongoDB Inc. alongside the server. There is no
plausible alternative: the MongoDB wire protocol is a binary BSON
protocol over a custom framing, not something to reimplement over httpx
the way this package's Elasticsearch connector reimplements a plain
JSON REST API. `pymongo` alone is synchronous and would block the event
loop on every round trip; `motor` is the same driver with an async
surface, and it brings the BSON codec (`bson`) that this module needs
anyway to hand JSON-safe results back to callers.

**Known migration debt, stated up front.** MongoDB has folded async
support into `pymongo` itself (`pymongo.AsyncMongoClient`) and has put
`motor` into maintenance with an announced end of life. This connector
uses `motor` today because that is what is installed and verified
working in this venv, but the surface it actually touches -
`AsyncIOMotorClient`, `list_collection_names`, `find_one`, `find`,
`insert_one`/`update_one`/`delete_one`, `admin.command("ping")` - is
identical in name and signature on `AsyncMongoClient`, so the migration
is an import swap plus a re-run of this module's tests, not a rewrite.
Nothing else in Praxis imports `motor`.

**The schema problem, stated honestly.** MongoDB has no schema. Every
document in a collection may have entirely different fields, of
entirely different types, and the server has no catalogue to ask.
`describe()` therefore *samples* - one document per collection - and
what it returns is a description of that one document, not of the
collection. A collection where 1% of documents carry an `error_details`
sub-document will, 99 times out of 100, be described as not having one.
The returned schema labels this (`"inferred_from": "one sampled
document"`) rather than presenting it as authoritative, because the
consumer of `describe()` in this codebase is
`praxis.agents.capability_factory.CapabilityFactory`, which puts it
into a prompt that generates code - and a model told "these are the
fields" will generate code that assumes it.

Collections with schema validators configured (`$jsonSchema`) *do*
have a real schema server-side; reading it is a separate
`listCollections` detail this connector does not fetch, and a
deployment that relies on validators will find `describe()` less
informative than its data actually warrants.
"""
from __future__ import annotations

import json
from typing import Any

from bson import json_util
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import (
    ConfigurationError,
    ConnectionFailure,
    InvalidOperation,
    OperationFailure,
    PyMongoError,
)

from praxis.connectors.errors import (
    ConnectorConfigurationError,
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

# Short on purpose. pymongo's own default is 30s, which is tuned for an
# application that must ride out a replica-set election; a connector
# inside an agent's execution step would rather fail in five seconds
# with "unreachable" than stall a task for half a minute.
_DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 5000

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000

# `describe()` does one round trip per collection. A database with
# hundreds of collections would otherwise turn a single `describe()`
# into hundreds of sequential queries; past this many the result is
# marked truncated rather than silently cut.
_MAX_DESCRIBED_COLLECTIONS = 50

# Refused inside a read-only connector's filter. None of these is a
# *write* - MongoDB's server-side JavaScript cannot modify data - so
# this is not a mutation guard. It is refused because all three execute
# caller-supplied JavaScript inside the server process: unindexed by
# construction, unbounded in cost, and exactly the kind of thing a
# read-only posture is supposed to mean "no" to. A deployment that
# genuinely needs `$where` can register the connector with
# `read_only=False`, which makes that choice explicit and auditable.
_SERVER_SIDE_JS_OPERATORS = frozenset({"$where", "$function", "$accumulator"})

_BSON_TYPE_NAMES = {
    "ObjectId": "objectId",
    "datetime": "date",
    "Decimal128": "decimal128",
    "Binary": "binary",
    "dict": "object",
    "list": "array",
    "str": "string",
    "int": "int",
    "float": "double",
    "bool": "bool",
    "NoneType": "null",
}


class MongoDBConnector(Connector):
    """Read (and, if configured, write) access to one MongoDB database.

    A new `AsyncIOMotorClient` is built per operation and closed again,
    rather than held for the connector's lifetime. That is a real cost
    - each call pays connection setup - and it is paid deliberately:
    motor binds a client to the event loop that was running when the
    client was created, and `Connector` has no `close()` in its
    contract for a long-lived client to be torn down through. A client
    cached on the instance would therefore outlive its loop the first
    time a connector built at import time is used from a second loop
    (which is precisely what this repo's own test suite does, one loop
    per test) and fail with an opaque "attached to a different loop"
    error. The same reasoning is why
    `praxis.connectors.sql.sql_connector.SQLConnector` creates and
    disposes a SQLAlchemy engine per call.
    """

    def __init__(
        self,
        uri: str,
        *,
        database: str | None = None,
        name: str = "mongodb",
        read_only: bool = True,
        server_selection_timeout_ms: int = _DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self._uri = uri
        self._database_name = database
        self._server_selection_timeout_ms = server_selection_timeout_ms

    def _client(self) -> AsyncIOMotorClient:
        return AsyncIOMotorClient(
            self._uri, serverSelectionTimeoutMS=self._server_selection_timeout_ms
        )

    def _database(self, client: AsyncIOMotorClient) -> AsyncIOMotorDatabase:
        """The configured database, or the one named in the URI's path.

        Resolved here rather than in `__init__` because working it out
        from the URI alone requires parsing it, and parsing a
        `mongodb+srv://` URI performs a live DNS SRV lookup - an
        `__init__` that does network I/O would make
        `build_registry()` hang on a bad DNS server at startup.
        """
        if self._database_name:
            return client[self._database_name]
        try:
            return client.get_default_database()
        except ConfigurationError as exc:
            raise ConnectorConfigurationError(
                f"connector '{self.name}' has no database to use",
                connector_name=self.name,
                detail=(
                    "set PRAXIS_MONGODB_DATABASE, or put a database in the URI path "
                    "(mongodb://host:27017/<database>)"
                ),
            ) from exc

    async def describe(self) -> ConnectorDescription:
        """Lists collections and, per collection, the fields of one sampled document.

        See this module's docstring for why "one sampled document" is
        the honest description of what this returns and why it must not
        be read as a schema. An empty collection yields an empty field
        map, which is accurate - there is genuinely nothing to sample.
        """
        client = self._client()
        try:
            database = self._database(client)
            collection_names = sorted(await database.list_collection_names())
            truncated = len(collection_names) > _MAX_DESCRIBED_COLLECTIONS
            sampled_names = collection_names[:_MAX_DESCRIBED_COLLECTIONS]

            collections: dict[str, dict[str, str]] = {}
            for collection_name in sampled_names:
                document = await database[collection_name].find_one()
                collections[collection_name] = _infer_fields(document)
        except PyMongoError as exc:
            raise _map_mongo_error(exc, self.name, "describe") from exc
        finally:
            client.close()

        return ConnectorDescription(
            kind="mongodb",
            schema={
                "database": self._database_name or "(from URI)",
                "read_only": self.read_only,
                "inferred_from": "one sampled document per collection; not a schema",
                "collection_count": len(collection_names),
                "truncated": truncated,
                "collections": collections,
            },
        )

    async def read(self, query: str, **params: Any) -> Any:
        """Runs a find, described by `query`: a JSON object string.

        Shape::

            {"collection": "orders",
             "filter": {"status": "open"},
             "limit": 50,
             "projection": {"_id": 0, "total": 1},
             "sort": [["created_at", -1]]}

        Only `collection` is required. `filter` defaults to `{}` (every
        document), `limit` to 100 and is refused above 1000, and
        `projection`/`sort` are optional. `sort` is a list of
        `[field, direction]` pairs rather than an object because JSON
        objects have no guaranteed key order and a multi-key sort's
        order is the whole point.

        This is deliberately a `find`, not an `aggregate`. An
        aggregation pipeline can end in `$out` or `$merge`, which write
        entire collections - allowing pipelines through a method named
        `read` would mean the read-only guard had to understand every
        stage of a pipeline language that gains new stages every server
        release. `find` cannot write, at all, by construction; that is
        a property of the operation rather than of a check this
        connector performs, which makes it worth giving up aggregations
        for.

        Results come back as Extended JSON (relaxed mode): an
        `ObjectId` becomes `{"$oid": "..."}`, a `datetime` becomes an
        ISO-8601 string. Raw BSON values are not JSON-serialisable, and
        everything downstream of a connector here - a task result row,
        an LLM prompt, an API response - has to serialise. Extended
        JSON is MongoDB's own standard for this and is lossless, unlike
        a `str()` of each exotic value.
        """
        if params:
            raise ConnectorQueryError(
                f"connector '{self.name}' read takes its whole request in the query string",
                connector_name=self.name,
                detail=f"unexpected keyword arguments: {sorted(params)}",
            )

        request = _parse_read_request(query, self.name)
        collection_name = request["collection"]
        query_filter = request["filter"]

        if self.read_only:
            offending = _find_server_side_js(query_filter)
            if offending:
                raise ConnectorReadOnlyError(
                    f"connector '{self.name}' is registered read-only; refusing a filter "
                    f"using the server-side JavaScript operator '{offending}'",
                    connector_name=self.name,
                )

        client = self._client()
        try:
            database = self._database(client)
            cursor = database[collection_name].find(query_filter, request["projection"])
            if request["sort"] is not None:
                cursor = cursor.sort(request["sort"])
            documents = await cursor.limit(request["limit"]).to_list(length=request["limit"])
        except PyMongoError as exc:
            raise _map_mongo_error(exc, self.name, f"find on '{collection_name}'") from exc
        finally:
            client.close()

        return _to_extended_json(documents)

    async def write(self, action: str, **params: Any) -> Any:
        """`action` is `"insert_one"`, `"update_one"` or `"delete_one"`.

        Every action requires `collection`. `insert_one` requires
        `document`; `update_one` requires `filter` and `update` (an
        update document using operators such as `$set`); `delete_one`
        requires `filter`.

        Singular by design. `update_many`/`delete_many` differ from
        these only in a keyword, and that keyword is the difference
        between "this changed one record" and "this changed the
        collection" - a distinction that should be made by an operator
        extending this connector for a specific need, not offered by
        default to generated code. `update_one` also does not upsert.
        """
        if self.read_only:
            raise ConnectorReadOnlyError(
                f"connector '{self.name}' is registered read-only; refusing action '{action}'",
                connector_name=self.name,
            )

        if action not in ("insert_one", "update_one", "delete_one"):
            raise NotImplementedError(
                f"unsupported action '{action}' for connector '{self.name}' "
                "(expected 'insert_one', 'update_one' or 'delete_one')"
            )

        collection_name = params.pop("collection", None)
        if not isinstance(collection_name, str) or not collection_name:
            raise ConnectorQueryError(
                f"connector '{self.name}' action '{action}' requires a non-empty 'collection'",
                connector_name=self.name,
            )

        client = self._client()
        try:
            collection = self._database(client)[collection_name]
            if action == "insert_one":
                document = _require_mapping(
                    params.pop("document", None), "document", self.name, action
                )
                _reject_extra(params, self.name, action, "collection/document")
                inserted = await collection.insert_one(document)
                return {"inserted_id": str(inserted.inserted_id)}

            if action == "update_one":
                query_filter = _require_mapping(
                    params.pop("filter", None), "filter", self.name, action
                )
                update = _require_mapping(params.pop("update", None), "update", self.name, action)
                _reject_extra(params, self.name, action, "collection/filter/update")
                updated = await collection.update_one(query_filter, update)
                return {
                    "matched_count": updated.matched_count,
                    "modified_count": updated.modified_count,
                }

            # Only "delete_one" reaches here: the action set was checked
            # against the three supported names above, before a client
            # was even built.
            query_filter = _require_mapping(params.pop("filter", None), "filter", self.name, action)
            _reject_extra(params, self.name, action, "collection/filter")
            deleted = await collection.delete_one(query_filter)
            return {"deleted_count": deleted.deleted_count}
        except PyMongoError as exc:
            raise _map_mongo_error(exc, self.name, f"{action} on '{collection_name}'") from exc
        finally:
            client.close()

    async def health(self) -> HealthStatus:
        """Runs the `ping` admin command, which every MongoDB deployment
        answers and which needs no privileges beyond connecting.

        Deliberately `ping` on the `admin` database rather than a read
        against the configured database: a credential scoped to one
        database still answers `ping`, so this reports "the server is
        up and I can authenticate" without also requiring read grants
        that a health check has no business needing.
        """
        client = self._client()
        try:
            await client.admin.command("ping")
            return HealthStatus(name=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))
        finally:
            client.close()


def _parse_read_request(query: str, connector_name: str) -> dict[str, Any]:
    """Validates and normalises the JSON request `read()` takes.

    Every field is checked and every rejection names what was wrong.
    The alternative - coercing whatever arrived, e.g. treating a
    non-integer `limit` as the default - would turn a caller's typo
    into a silently different query, which is the exact failure mode
    that makes a wrong answer look like a right one.
    """
    try:
        request = json.loads(query)
    except (TypeError, ValueError) as exc:
        raise ConnectorQueryError(
            f"connector '{connector_name}' read expects a JSON object string",
            connector_name=connector_name,
            detail=describe_exception(exc),
        ) from exc

    if not isinstance(request, dict):
        raise ConnectorQueryError(
            f"connector '{connector_name}' read expects a JSON object",
            connector_name=connector_name,
            detail=f"got {type(request).__name__}",
        )

    collection = request.get("collection")
    if not isinstance(collection, str) or not collection:
        raise ConnectorQueryError(
            f"connector '{connector_name}' read requires a non-empty 'collection'",
            connector_name=connector_name,
            detail=f"got {collection!r}",
        )

    query_filter = request.get("filter", {})
    if not isinstance(query_filter, dict):
        raise ConnectorQueryError(
            f"connector '{connector_name}' read expects 'filter' to be an object",
            connector_name=connector_name,
            detail=f"got {type(query_filter).__name__}",
        )

    projection = request.get("projection")
    if projection is not None and not isinstance(projection, dict):
        raise ConnectorQueryError(
            f"connector '{connector_name}' read expects 'projection' to be an object",
            connector_name=connector_name,
            detail=f"got {type(projection).__name__}",
        )

    sort = request.get("sort")
    if sort is not None:
        if not isinstance(sort, list) or not all(
            isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str) for pair in sort
        ):
            raise ConnectorQueryError(
                f"connector '{connector_name}' read expects 'sort' to be a list of "
                "[field, direction] pairs",
                connector_name=connector_name,
                detail=f"got {sort!r}",
            )
        sort = [(field, direction) for field, direction in sort]

    limit = request.get("limit", _DEFAULT_LIMIT)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ConnectorQueryError(
            f"connector '{connector_name}' read expects 'limit' to be a positive integer",
            connector_name=connector_name,
            detail=f"got {limit!r}",
        )
    if limit > _MAX_LIMIT:
        raise ConnectorQueryError(
            f"connector '{connector_name}' refuses a find with limit={limit}",
            connector_name=connector_name,
            detail=(
                f"the per-query bound is {_MAX_LIMIT}; narrow the filter rather than "
                "pulling a large collection through this connector"
            ),
        )

    unknown = set(request) - {"collection", "filter", "limit", "projection", "sort"}
    if unknown:
        raise ConnectorQueryError(
            f"connector '{connector_name}' read got unknown request keys",
            connector_name=connector_name,
            detail=(
                f"unrecognised: {sorted(unknown)}; expected "
                "collection/filter/limit/projection/sort"
            ),
        )

    return {
        "collection": collection,
        "filter": query_filter,
        "limit": limit,
        "projection": projection,
        "sort": sort,
    }


def _find_server_side_js(value: Any) -> str | None:
    """The first server-side-JavaScript operator found anywhere in `value`, or None.

    Recursive because these operators are legal at any nesting depth -
    `{"$and": [{"$where": "..."}]}` is a perfectly valid filter, and a
    check that only looked at top-level keys would miss it entirely.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _SERVER_SIDE_JS_OPERATORS:
                return key
            found = _find_server_side_js(nested)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_server_side_js(item)
            if found:
                return found
    return None


def _infer_fields(document: Any) -> dict[str, str]:
    """Top-level field names of one document, with a BSON-ish type name each.

    Top level only: a nested document's own fields are reported as the
    single type `object`. Recursing would make `describe()` output grow
    with document depth for a payoff that a *one-document sample*
    cannot justify - the nested shape would be no more representative
    of the collection than the top level already is.
    """
    if not isinstance(document, dict):
        return {}
    return {
        field: _BSON_TYPE_NAMES.get(type(value).__name__, type(value).__name__)
        for field, value in document.items()
    }


def _to_extended_json(documents: list[Any]) -> list[Any]:
    """BSON documents -> plain JSON-safe Python via Extended JSON (relaxed).

    Round-tripping through `json_util.dumps`/`json.loads` rather than
    walking the documents by hand: `bson` already knows every BSON type
    and its canonical JSON form, including the ones easy to forget
    (`Decimal128`, `Binary` subtypes, `DBRef`, `Timestamp`), and a
    hand-rolled walker would silently stringify whichever ones it had
    not anticipated.
    """
    return json.loads(json_util.dumps(documents, json_options=json_util.RELAXED_JSON_OPTIONS))


def _require_mapping(value: Any, field: str, connector_name: str, action: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConnectorQueryError(
            f"connector '{connector_name}' action '{action}' requires '{field}' to be an object",
            connector_name=connector_name,
            detail=f"got {type(value).__name__}",
        )
    return value


def _reject_extra(params: dict[str, Any], connector_name: str, action: str, expected: str) -> None:
    if params:
        raise ConnectorQueryError(
            f"connector '{connector_name}' action '{action}' got unexpected arguments",
            connector_name=connector_name,
            detail=f"unrecognised: {sorted(params)}; expected {expected}",
        )


def _map_mongo_error(exc: PyMongoError, connector_name: str, operation: str) -> Exception:
    """pymongo's exception tree -> this package's three-way taxonomy.

    `ConnectionFailure` is the base of `ServerSelectionTimeoutError`,
    `NetworkTimeout` and `AutoReconnect`, which together cover "the
    server is down, unroutable, or too slow" - all retryable, so all
    `ConnectorUnavailableError`. `OperationFailure` is the server
    having answered with a refusal (bad query operator, unauthorized,
    unknown index hint) and carries a numeric server error code worth
    keeping. `ConfigurationError` is a malformed URI or an impossible
    option combination: nothing was sent and no retry helps.
    """
    if isinstance(exc, ConnectionFailure):
        return ConnectorUnavailableError(
            f"connector '{connector_name}' could not reach MongoDB for {operation}",
            connector_name=connector_name,
            detail=describe_exception(exc),
        )
    if isinstance(exc, ConfigurationError):
        return ConnectorConfigurationError(
            f"connector '{connector_name}' is misconfigured for {operation}",
            connector_name=connector_name,
            detail=describe_exception(exc),
        )
    if isinstance(exc, OperationFailure):
        return ConnectorQueryError(
            f"connector '{connector_name}' had {operation} refused by MongoDB",
            connector_name=connector_name,
            detail=describe_exception(exc),
            status=exc.code,
        )
    if isinstance(exc, InvalidOperation):
        return ConnectorQueryError(
            f"connector '{connector_name}' attempted an invalid {operation}",
            connector_name=connector_name,
            detail=describe_exception(exc),
        )
    return ConnectorQueryError(
        f"connector '{connector_name}' failed {operation}",
        connector_name=connector_name,
        detail=f"{type(exc).__name__}: {exc}",
    )


register_connector_factory(
    ConnectorFactory(
        name="mongodb",
        is_configured=lambda settings: bool(settings.mongodb_uri),
        build=lambda settings: MongoDBConnector(
            uri=required(settings.mongodb_uri, setting="mongodb_uri"),
            database=settings.mongodb_database,
        ),
    )
)
