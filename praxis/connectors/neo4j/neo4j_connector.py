# praxis/connectors/neo4j/neo4j_connector.py
"""Neo4j connector, built on the official async driver (spec §6).

Neo4j speaks Bolt - a binary, packstream-framed protocol over TCP with
its own handshake, chunking and version negotiation - so there is no
httpx-shaped alternative here the way there is for Elasticsearch. Neo4j
does ship an HTTP API, but it is the deprecated path, lacks routing for
clustered deployments, and would still need the same Cypher handling on
top. `neo4j` is the vendor's own driver, it has first-class asyncio
support (`AsyncGraphDatabase`), and it is what the `neo4j://` routing
scheme requires to talk to a cluster at all.

**The read-only guarantee here is stronger than in the SQL connectors,
and the reason is worth knowing.** `PostgresConnector` and
`SQLConnector` enforce read-only with a client-side check (a regex, and
in `SQLConnector`'s case a real `sqlglot` parse) because the wire
protocol has no notion of a read-only statement. Bolt does: a
transaction is opened in READ or WRITE access mode, and a READ
transaction is refused *by the server* the moment anything in it tries
to write - including from inside a stored procedure, which no
client-side parse could ever see into. `read()` therefore runs inside
`session.execute_read`, and that, not the clause scan below, is what
actually makes it read-only. The clause scan exists to turn "the server
rejected your transaction after a round trip" into a clear local
refusal naming the offending clause, and to catch the case before any
data is touched.
"""
from __future__ import annotations

import contextlib
import re
from typing import Any

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ConfigurationError, DriverError, Neo4jError
from neo4j.graph import Node, Path, Relationship
from neo4j.spatial import Point
from neo4j.time import Date, DateTime, Duration, Time

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

_DEFAULT_MAX_RECORDS = 1000
_DEFAULT_TIMEOUT_SECONDS = 30.0

# The driver's own default here is 30 seconds: `execute_read`/
# `execute_write` keep re-running the whole transaction function for
# that long whenever the failure looks retryable, which includes "the
# server is not there". Praxis already has a bounded retry a level up
# (`praxis.connectors.retry.call_with_retry`, 3 attempts), so leaving
# the driver's default in place would compose into a task step that
# blocks for a minute and a half before reporting a host that was never
# listening. Ten seconds is long enough to ride out a cluster leader
# re-election, which is the case the driver's retry actually exists for.
_DEFAULT_TRANSACTION_RETRY_SECONDS = 10.0

# Cypher's write clauses. Unlike SQL, these are not restricted to the
# start of a statement - `MATCH (n:User) SET n.active = false` is a
# perfectly ordinary write whose first word is a read clause - so this
# is a whole-query word search rather than a prefix match.
#
# `LOAD CSV` and `FOREACH` are included for a different reason than the
# rest: neither writes by itself, but both exist almost exclusively to
# drive writes, and both can pull unbounded external data into a query.
_WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b",
    re.IGNORECASE,
)

# Comments and every kind of quoted text, stripped before the clause
# scan so that a node property literally containing the word "delete",
# or a backtick-quoted label called `SET`, cannot be mistaken for a
# clause. Ordered so that block comments are consumed before anything
# inside them is looked at.
_CYPHER_LITERALS = re.compile(
    r"/\*.*?\*/"          # /* block comment */
    r"|//[^\n]*"          # // line comment
    r"|'(?:\\.|[^'\\])*'"  # 'single quoted'
    r"|\"(?:\\.|[^\"\\])*\""  # "double quoted"
    r"|`(?:[^`]|``)*`",   # `backtick quoted identifier`
    re.DOTALL,
)

# `db.labels()`/`db.relationshipTypes()`/`db.propertyKeys()` rather than
# Neo4j 5's `SHOW` commands: these three procedures work unchanged on
# 4.x and 5.x, and this connector has no way to know which major version
# an operator has pointed it at.
_DESCRIBE_QUERIES = {
    "labels": "CALL db.labels() YIELD label RETURN label ORDER BY label",
    "relationship_types": (
        "CALL db.relationshipTypes() YIELD relationshipType "
        "RETURN relationshipType ORDER BY relationshipType"
    ),
    "property_keys": (
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey"
    ),
}


class Neo4jConnector(Connector):
    """Read (and, if configured, write) access to one Neo4j database.

    A driver is created and closed per operation rather than held for
    the connector's lifetime. This is the same event-loop-affinity
    trade-off documented on
    `praxis.connectors.mongodb.mongodb_connector.MongoDBConnector`, and
    it costs more here than there: a Neo4j driver owns a connection
    pool, so per-call creation means a fresh Bolt handshake per call
    rather than a pooled connection. It is still the right default
    while `Connector` has no `close()` in its contract - a pool with no
    shutdown hook leaks sockets and, worse, silently binds to whichever
    event loop happened to be running at import time. A caller running
    many queries in one loop and willing to manage the lifetime should
    use the driver directly rather than have this connector cache one.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str | None = None,
        name: str = "neo4j",
        read_only: bool = True,
        max_records: int = _DEFAULT_MAX_RECORDS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        transaction_retry_seconds: float = _DEFAULT_TRANSACTION_RETRY_SECONDS,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self._uri = uri
        self._auth = (user, password)
        self._database = database
        self._max_records = max_records
        self._timeout_seconds = timeout_seconds
        self._transaction_retry_seconds = transaction_retry_seconds

    def _driver(self) -> Any:
        return AsyncGraphDatabase.driver(
            self._uri,
            auth=self._auth,
            connection_acquisition_timeout=self._timeout_seconds,
            max_transaction_retry_time=self._transaction_retry_seconds,
        )

    async def describe(self) -> ConnectorDescription:
        """Returns the database's node labels, relationship types and property keys.

        **These are catalogue-wide, not correlated.** Neo4j records
        every label, type and property key that has *ever* existed in
        the database, and does not record which properties belong to
        which label - so this says "the graph contains `:Person` nodes
        and a `since` property somewhere", never "a `:Person` has a
        `since`". Property keys also survive the deletion of everything
        that used them until the store is compacted, so a key listed
        here may match nothing at all today.

        Correlating them is possible (`CALL db.schema.nodeTypeProperties()`
        on Enterprise, or an `apoc.meta.schema` scan) but both options
        either require a specific edition or walk the whole store, and
        neither is something a `describe()` should do to a production
        graph without being asked.
        """
        driver = self._driver()
        try:
            async with driver.session(database=self._database) as session:
                schema: dict[str, Any] = {
                    "database": self._database or "(server default)",
                    "read_only": self.read_only,
                    "correlated": False,
                }
                for key, query in _DESCRIBE_QUERIES.items():
                    records = await session.execute_read(
                        _collect, query, {}, self._max_records, self.name
                    )
                    schema[key] = [next(iter(record.values())) for record in records]
        except (Neo4jError, DriverError, OSError) as exc:
            raise _map_neo4j_error(exc, self.name, "describe") from exc
        finally:
            await driver.close()

        return ConnectorDescription(kind="neo4j", schema=schema)

    async def read(self, query: str, **params: Any) -> Any:
        """Runs `query` as Cypher inside a READ transaction and returns its records.

        Keyword arguments become Cypher parameters, so
        ``read("MATCH (p:Person {name: $name}) RETURN p", name="Ada")``
        binds `$name` properly rather than by string interpolation -
        which also means this connector never builds a query by
        concatenation and is not injectable through its parameters.

        The result is bounded at `max_records` (1000 by default), and
        exceeding it **raises** rather than truncating. Truncation is
        the wrong failure for a graph query in particular: a partial
        traversal is not a smaller correct answer, it is a different
        and wrong one, and a caller handed 1000 of 50000 paths has no
        way to tell. The fix is a `LIMIT` in the query, which the error
        says.

        Driver-native values are converted to JSON-safe equivalents -
        see `_jsonable`.
        """
        if self.read_only:
            offending = _find_write_clause(query)
            if offending:
                raise ConnectorReadOnlyError(
                    f"connector '{self.name}' is registered read-only; refusing a Cypher "
                    f"query containing the write clause '{offending}'",
                    connector_name=self.name,
                )

        driver = self._driver()
        try:
            async with driver.session(database=self._database) as session:
                records = await session.execute_read(
                    _collect, query, params, self._max_records, self.name
                )
        except ConnectorQueryError:
            # Raised by `_collect` itself when the record bound is hit -
            # already a taxonomy error, so it must not be re-mapped by
            # the handler below into a different one.
            raise
        except (Neo4jError, DriverError, OSError) as exc:
            raise _map_neo4j_error(exc, self.name, "read") from exc
        finally:
            await driver.close()

        return [_jsonable(record.data()) for record in records]

    async def write(self, action: str, **params: Any) -> Any:
        """`action` must be `"cypher"`; `params["query"]` is the Cypher to run.

        Every other keyword argument becomes a Cypher parameter, the
        same as on `read()`. Runs inside `session.execute_write`, so
        the driver retries it on a transient failure (a leader
        re-election in a cluster) - which is safe precisely because
        Cypher writes are expressed as idempotent `MERGE`/`SET` far
        more often than not, but is worth knowing: a write expressed
        with `CREATE` can be applied twice if the first attempt's
        commit acknowledgement is lost.
        """
        if self.read_only:
            raise ConnectorReadOnlyError(
                f"connector '{self.name}' is registered read-only; refusing action '{action}'",
                connector_name=self.name,
            )
        if action != "cypher":
            raise NotImplementedError(
                f"unsupported action '{action}' for connector '{self.name}' (expected 'cypher')"
            )

        query = params.pop("query", None)
        if not isinstance(query, str) or not query.strip():
            raise ConnectorQueryError(
                f"connector '{self.name}' action 'cypher' requires a non-empty 'query'",
                connector_name=self.name,
            )

        driver = self._driver()
        try:
            async with driver.session(database=self._database) as session:
                records = await session.execute_write(
                    _collect, query, params, self._max_records, self.name
                )
        except ConnectorQueryError:
            raise
        except (Neo4jError, DriverError, OSError) as exc:
            raise _map_neo4j_error(exc, self.name, "write") from exc
        finally:
            await driver.close()

        return [_jsonable(record.data()) for record in records]

    async def health(self) -> HealthStatus:
        """`driver.verify_connectivity()` - the driver's own check, which
        opens a connection, completes the Bolt handshake and
        authenticates, without running a query or touching a database.

        That last part matters: a query-based check would need read
        grants on some database and would fail on a credential that is
        perfectly valid but scoped elsewhere.
        """
        driver = self._driver()
        try:
            await driver.verify_connectivity()
            return HealthStatus(name=self.name, healthy=True)
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return HealthStatus(name=self.name, healthy=False, detail=describe_exception(exc))
        finally:
            # A driver whose connectivity check just failed may never
            # have opened a pool at all, and closing one can raise in
            # its own right - which would escape straight past the
            # `except` above and out of a method contracted never to
            # raise, purely as a cleanup side effect.
            with contextlib.suppress(Exception):
                await driver.close()


async def _collect(
    tx: Any,
    query: str,
    parameters: dict[str, Any],
    max_records: int,
    connector_name: str,
) -> list[Any]:
    """Transaction function: run `query` and materialise up to `max_records` records.

    A plain function rather than a lambda because `execute_read`/
    `execute_write` may call it more than once - the driver retries the
    whole unit of work on a transient error - so it has to be
    re-runnable from the top, which a closure over a partially-consumed
    result would not be.
    """
    result = await tx.run(query, parameters)
    records: list[Any] = []
    async for record in result:
        if len(records) >= max_records:
            raise ConnectorQueryError(
                "query returned more records than this connector will materialise",
                connector_name=connector_name,
                detail=(
                    f"the bound is {max_records} records; add an explicit LIMIT to the query "
                    "rather than receiving a silently truncated traversal"
                ),
            )
        records.append(record)
    return records


def _find_write_clause(query: str) -> str | None:
    """The first Cypher write clause in `query`, ignoring comments and literals.

    See this module's docstring: this is a local, fast refusal, not the
    actual read-only enforcement (the server's READ transaction mode
    is). It deliberately errs toward refusing - a query that merely
    mentions `SET` outside a literal is rejected rather than sent -
    because a false refusal costs a caller one clear error message and
    a false acceptance costs a round trip to discover the same thing.
    """
    stripped = _CYPHER_LITERALS.sub(" ", query)
    match = _WRITE_CLAUSES.search(stripped)
    return match.group(0).upper() if match else None


def _jsonable(value: Any) -> Any:
    """Driver-native values -> JSON-serialisable equivalents.

    `Record.data()` already flattens nodes and relationships to their
    property dicts, but it leaves Neo4j's own temporal and spatial
    types in place, and those are not JSON-serialisable - a task result
    carrying one would fail at the API boundary, far from here, with an
    error naming neither this connector nor the offending property.

    Temporal values become their ISO-8601 form (Neo4j's `Duration` has
    no Python equivalent at all - `P3DT4H` is not a `timedelta`, since
    months and days are calendar-relative - so ISO-8601 is the only
    lossless representation). A `Point` becomes its SRID plus
    coordinates rather than a WKT string, because the SRID is what says
    whether those coordinates are cartesian or lat/long.
    """
    if isinstance(value, (Date, Time, DateTime, Duration)):
        return value.iso_format()
    if isinstance(value, Point):
        return {"srid": value.srid, "coordinates": list(value)}
    if isinstance(value, (Node, Relationship)):
        return {key: _jsonable(item) for key, item in dict(value).items()}
    if isinstance(value, Path):
        return {
            "nodes": [_jsonable(node) for node in value.nodes],
            "relationships": [_jsonable(relationship) for relationship in value.relationships],
        }
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _map_neo4j_error(exc: Exception, connector_name: str, operation: str) -> Exception:
    """The driver's exception tree -> this package's three-way taxonomy.

    The driver's own split is the useful one and is followed here:
    `Neo4jError` is the *server* having rejected something (it carries
    a structured `code` like
    `Neo.ClientError.Statement.SyntaxError`), while `DriverError` is
    the client never having got that far - `ServiceUnavailable`,
    `SessionExpired`, a routing failure.

    `ConfigurationError` is the one `DriverError` that is not a
    transport problem (a malformed URI, an impossible option
    combination), so it is split out as a configuration fault where no
    retry can help.
    """
    if isinstance(exc, ConfigurationError):
        return ConnectorConfigurationError(
            f"connector '{connector_name}' is misconfigured for {operation}",
            connector_name=connector_name,
            detail=describe_exception(exc),
        )
    if isinstance(exc, Neo4jError):
        return ConnectorQueryError(
            f"connector '{connector_name}' had its {operation} rejected by Neo4j",
            connector_name=connector_name,
            detail=describe_exception(exc),
            status=getattr(exc, "code", None),
        )
    if isinstance(exc, (DriverError, OSError)):
        return ConnectorUnavailableError(
            f"connector '{connector_name}' could not reach Neo4j for {operation}",
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
        name="neo4j",
        is_configured=lambda settings: bool(
            settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password
        ),
        build=lambda settings: Neo4jConnector(
            uri=required(settings.neo4j_uri, setting="neo4j_uri"),
            user=required(settings.neo4j_user, setting="neo4j_user"),
            password=required(settings.neo4j_password, setting="neo4j_password"),
            database=settings.neo4j_database,
        ),
    )
)
