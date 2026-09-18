"""Deployment-wide settings (spec §2; §19 step 2 - config validation)."""
from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseModel):
    """One entry in ``Settings.mcp_servers``: everything needed to build one
    ``MCPConnector`` (spec §6's "register it as an MCP server ... no core
    code change").

    A pydantic ``BaseModel`` (not a dataclass) so pydantic-settings can
    parse a whole list of these straight out of a JSON-encoded env var.
    Exactly one of ``command`` (stdio) or ``url`` (HTTP/SSE) is expected
    to be set per entry - the same validation
    ``praxis.connectors.mcp.mcp_connector.MCPConnector.__init__`` enforces
    when `praxis.connectors.bootstrap.build_registry` constructs one
    from this config.
    """

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    # Tool names this server offers that are known-safe to call through
    # the read-only path, for servers that do not annotate their tools.
    #
    # MCP lets a server declare `readOnlyHint` per tool, and when it
    # does, that declaration is used and this list is unnecessary. When
    # it does not, a generic "call any tool" read path would let a
    # mutating tool bypass the Orchestrator's approval gate entirely
    # (`query_connector` is itself risk=read_only). So an unannotated
    # tool is refused unless an operator names it here - the decision
    # is then explicit and visible in config rather than implicit.
    read_only_tools: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRAXIS_", env_file=".env", extra="ignore")

    database_url: str = Field(
        ...,
        description="Async SQLAlchemy URL, e.g. postgresql+asyncpg://user:pass@host/db",
    )
    docker_host: str | None = Field(
        default=None, description="Override for the Docker daemon socket"
    )
    sandbox_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Phase 7 (Observability, spec §18): "Scheduled health scan
    # (APScheduler, e.g. every 5 min) writes health history to Postgres."
    # Wired into `praxis.agents.scheduler.Scheduler` by
    # `praxis.api.main`'s module-level setup.
    health_scan_interval_seconds: int = Field(default=300, ge=1)

    # How often the application looks for due user schedules
    # (`praxis.api.main.run_due_schedules`). Sixty seconds is a
    # deliberate floor on granularity: a schedule is due at a
    # minute boundary at finest, so polling faster only costs
    # queries. Due-ness lives in a column, so a poll missed
    # during a restart is picked up by the next one rather than
    # being lost.
    schedule_poll_interval_seconds: int = Field(default=60, ge=1)

    # An extra directory of declarative SKILL.md procedures this
    # deployment supplies - typically a mounted volume. Loaded
    # after the ones that ship inside the package, so an operator
    # can add capabilities without rebuilding an image.
    procedures_dir: str | None = Field(
        default=None,
        description="Additional directory to scan for SKILL.md procedures",
    )

    # Phase 12 (Multi-tenancy, identity, RBAC/ABAC - Prompt §8, §11).
    # Off by default so a single-operator deployment (and this repo's
    # own test suite) keeps working exactly as before: with auth
    # disabled every request runs as `praxis.security.principal.
    # SYSTEM_PRINCIPAL`, inside the real `DEFAULT_TENANT_ID` tenant, and
    # still flows through the identical `PolicyEngine` + audit path - the
    # isolation code is never bypassed, only the *credential* step is.
    # Switching this on makes `X-API-Key` (or `Authorization: Bearer`)
    # mandatory on every non-public endpoint.
    auth_enabled: bool = Field(
        default=False,
        description="Require an API key on every request and resolve a real per-user Principal",
    )

    # How long a pending approval stays decidable before the sweep fails
    # the task (Prompt §8's "expiry" binding). Distinct from
    # `approval_timeout_seconds` below, which is when the *task* is
    # swept; this is the window written onto the `ApprovalRecord` itself,
    # and defaults to the same value so the two never disagree unless an
    # operator deliberately sets them apart.
    approval_ttl_seconds: int = Field(default=3600, ge=1)

    # Phase 21 (Prompt §7: "Generated agents/tools must never be
    # silently trusted ... require approval before publishing/enabling
    # it").
    #
    # Defaults to TRUE, unlike most toggles here, and the asymmetry is
    # deliberate: every other optional feature defaults off because its
    # absence is merely a missing capability, whereas this one
    # defaults on because its absence is a missing *control*.
    # "LLM-authored code runs unreviewed unless an operator remembered
    # to switch review on" is precisely the posture the brief forbids,
    # so the unsafe direction has to be the one someone opts into
    # explicitly - and that choice is then visible in their config
    # rather than implicit in ours.
    #
    # With this on, a synthesized skill is recorded as
    # `pending_approval`: sandbox-validated, catalogued, and visible in
    # the approval queue, but refused at execution until a principal
    # holding `skill:approve` signs off on its exact code hash
    # (`praxis.agents.publication`).
    require_skill_approval: bool = Field(
        default=True,
        description="Require human approval before a synthesized skill may be executed",
    )

    # Whether a finished task keeps its LangGraph checkpoints.
    #
    # Defaults to retaining them, because `Orchestrator.replay_history`
    # is the brief's "replay/debug mode" and the run an operator most
    # wants to inspect is a *failed* one - deleting a task's history the
    # moment it fails destroys exactly the evidence the feature exists
    # to provide. The cost is honest and worth naming: checkpoint rows
    # accumulate, roughly one per superstep per task, so a deployment
    # running many short tasks should either set this False or prune on
    # a schedule via `Orchestrator.prune_task_history`.
    # How many times a failed plan is corrected and retried before the
    # task is reported as failed.
    #
    # A first plan is often wrong in a way the failure itself explains -
    # a query against a column that does not exist comes back naming the
    # columns that do - and stopping there turns a recoverable mistake
    # into a non-answer. Bounded because each retry is a real model call:
    # a plan still failing on the third attempt is usually blocked by
    # something more attempts cannot fix.
    max_replan_attempts: int = Field(
        default=2,
        ge=0,
        description="Times a failed plan may be re-planned with the failure as context",
    )

    # Whether a task that ran without errors must also be judged BY A
    # MODEL to have answered its intent before it is reported
    # completed. The completion gate's deterministic checks always run
    # regardless; this switches on the reviewer. Off by default because
    # on means one real provider call per finished task - a deployment
    # opts into that cost knowingly, and a test suite never pays it by
    # accident (with a provider key in the environment, the default-on
    # version made every API test that completed a task call Anthropic,
    # and the suite timed out).
    verify_completion: bool = Field(
        default=False,
        description="Judge a finished task's result against its intent before accepting it",
    )

    retain_checkpoints_after_completion: bool = Field(
        default=True,
        description="Keep a finished task's checkpoints so its run stays replayable",
    )

    # Per-connector resilience (brief §6's "rate limiting, retries,
    # circuit breakers"). See `praxis.connectors.resilience`.
    #
    # The circuit breaker defaults ON because it costs nothing on the
    # happy path - a closed circuit is one attribute check - and the
    # failure it prevents (retrying into a known-dead dependency, N
    # users turning one outage into 3N units of load) is real.
    connector_circuit_breaker_enabled: bool = Field(
        default=True,
        description="Open a per-connector circuit after repeated failures",
    )
    connector_failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive failures before a connector's circuit opens",
    )
    connector_circuit_reset_seconds: float = Field(
        default=30.0,
        gt=0,
        description="How long an open circuit waits before admitting one trial call",
    )
    # Rate limiting defaults OFF: a limit nobody asked for is a latency
    # bug waiting to be discovered in production, and the right value is
    # a property of the *vendor's* quota, which only the deployment
    # knows. Unset means unthrottled, which is the historical behaviour.
    connector_rate_limit_per_second: float | None = Field(
        default=None,
        gt=0,
        description="Optional per-connector call rate ceiling (calls per second)",
    )

    # Phase 7 (Exception handling, spec §12): how long a task may sit
    # `awaiting_approval`/`awaiting_clarification` before
    # `praxis.api.main.sweep_stale_approvals` marks it `failed` with an
    # `ApprovalTimeoutError` detail. Doubles as that sweep job's own
    # scheduling interval (spec's own suggested default, "e.g. 3600") -
    # a dedicated, separate sweep-interval setting would let the sweep
    # run far more often than the timeout it's checking for, which buys
    # nothing; running it once per timeout window is exactly often
    # enough to catch every task the moment it crosses that threshold.
    approval_timeout_seconds: int = Field(default=3600, ge=1)

    # Local-filesystem `BlobStore` root (spec §2: "Local filesystem
    # (scratch dir)" is the MVP-deployed backend). Has a default so its
    # absence never blocks Settings() construction the way a missing
    # database_url does - swapping to a different BlobStore backend
    # (S3, GCS, ...) per spec §2's fsspec-based swap path is providing a
    # different BlobStore implementation, not changing this setting's
    # meaning.
    blob_store_root: str = Field(
        default="./data/attachments",
        description="Local filesystem root directory for LocalBlobStore",
    )

    # Distributed cache (brief §10). Unset means the deployment uses
    # the in-process `InMemoryCache`, which is a supported
    # configuration, not a degraded one - `praxis.cache.redis_cache.
    # redis_cache_from_settings` returns None rather than raising, the
    # same "unset = simply off" rule the connector credentials below
    # follow. Setting it makes cache entries shared across workers and
    # makes `RedisCache.invalidate_prefix` able to reach entries this
    # process never wrote. Note that an unreachable Redis is NOT an
    # error either: `RedisCache` degrades every operation to a miss
    # (see its module docstring), so a cache outage costs the
    # deployment its speedup and nothing else.
    redis_url: str | None = Field(
        default=None,
        description="Redis URL (e.g. redis://localhost:6379/0) for the distributed Cache",
    )
    redis_namespace: str = Field(
        default="praxis",
        description="Key prefix isolating this deployment's entries in a shared Redis",
    )

    # S3-compatible object storage (spec §2's BlobStore swap path).
    # Unset means `LocalBlobStore` under `blob_store_root`. No
    # credential fields: botocore's standard chain (environment,
    # shared config, instance/task role) is how a deployment avoids
    # holding long-lived keys in application config at all, and adding
    # fields here would invite exactly that. `s3_endpoint_url` is what
    # points the same client at MinIO/R2/Ceph instead of AWS.
    s3_bucket: str | None = Field(
        default=None, description="Bucket name for S3BlobStore; unset means LocalBlobStore"
    )
    s3_endpoint_url: str | None = Field(
        default=None, description="Override endpoint for an S3-compatible service (MinIO, R2, ...)"
    )
    s3_region: str | None = Field(default=None, description="AWS region for S3BlobStore")
    s3_key_prefix: str = Field(
        default="",
        description="Bucket-relative prefix for every object; NOT a tenant boundary",
    )

    # Optional connector credentials (spec §6, §19 step 4). Each is
    # genuinely optional: an unset value means that connector is simply
    # not registered (praxis.connectors.bootstrap.build_registry skips
    # it, not fails) - the platform degrades gracefully, per spec §6.
    # No field here for the generic Postgres connector's DSN: connecting
    # to an arbitrary external Postgres DB (spec §16.2) is a per-task/
    # per-connector-registration concern, not a single global setting -
    # a settings field would wrongly imply there's only ever one.
    github_token: str | None = Field(
        default=None, description="GitHub personal access token for GitHubConnector"
    )
    slack_bot_token: str | None = Field(
        default=None, description="Slack bot token (xoxb-...) for SlackConnector"
    )
    prometheus_url: str | None = Field(
        default=None, description="Base URL of a Prometheus server for PrometheusConnector"
    )

    # Six more optional data-source connectors, on exactly the same terms
    # as the three above: every field here defaults to None, and an unset
    # field means `praxis.connectors.bootstrap.build_registry` simply does
    # not register that connector - never a validation error, never a
    # connector registered in a half-configured state that only fails at
    # query time. Each connector's own `is_configured` gate names the
    # minimum subset that makes it usable at all (e.g. neo4j needs all
    # three of uri/user/password; s3 needs both halves of the key pair),
    # so "registered" always means "has enough config to attempt a call".
    #
    # The optional extras alongside each minimum (a default database,
    # bucket, or auth header) are genuinely optional in a *different*
    # sense: their absence does not block registration, but it does
    # narrow what the connector can do, and each connector raises a
    # `ConnectorConfigurationError` naming the missing setting rather
    # than guessing - see `praxis.connectors.errors`.

    mongodb_uri: str | None = Field(
        default=None,
        description="MongoDB connection URI (mongodb:// or mongodb+srv://) for MongoDBConnector",
    )
    # Optional because a MongoDB URI may already name a default database
    # in its path (`mongodb://host:27017/analytics`). When it doesn't,
    # this is the only way to say which database to describe and query -
    # MongoDB has no "current database" the driver can infer.
    mongodb_database: str | None = Field(
        default=None,
        description="Database name for MongoDBConnector; overrides any database in mongodb_uri",
    )

    elasticsearch_url: str | None = Field(
        default=None,
        description="Base URL of an Elasticsearch/OpenSearch cluster for ElasticsearchConnector",
    )
    # Elastic's own recommended machine credential. Basic auth is
    # supported by the connector class (its `basic_auth` constructor
    # argument) but deliberately has no Settings field: an operator
    # wiring this up from config should be issuing a scoped API key, not
    # putting a cluster superuser's password in the environment.
    elasticsearch_api_key: str | None = Field(
        default=None,
        description="Base64 Elasticsearch API key (sent as 'Authorization: ApiKey ...')",
    )

    # S3-compatible object storage read as a DATA SOURCE (real AWS S3,
    # MinIO, Ceph RGW, Cloudflare R2, Backblaze B2's S3 endpoint, ...).
    #
    # Prefixed `s3_connector_` rather than reusing the plain `s3_*`
    # fields above, and the distinction is load-bearing, not cosmetic:
    # those belong to `S3BlobStore`, where `s3_bucket` means *Praxis's
    # own artifact bucket* - somewhere Praxis writes. Here `..._bucket`
    # means somebody else's data bucket that a task reads from. Sharing
    # one field would silently point a data connector at Praxis's own
    # artifact store, and a deployment that legitimately uses both
    # (artifacts in one bucket, customer data in another) could not
    # express that at all.
    #
    # Unlike `S3BlobStore`, this does take explicit credentials. That is
    # not a disagreement about whether botocore's ambient credential
    # chain is preferable - it is that the chain gives no way to answer
    # "is this connector configured?", and `build_registry` has to decide
    # whether to register at all *before* any call is made. The key pair
    # is therefore the registration gate; the other three refine where
    # and what.
    s3_connector_access_key_id: str | None = Field(
        default=None, description="Access key ID for S3Connector"
    )
    s3_connector_secret_access_key: str | None = Field(
        default=None, description="Secret access key for S3Connector"
    )
    # Unset means real AWS S3. Any other S3-compatible provider needs its
    # own endpoint here (e.g. http://localhost:9000 for a local MinIO).
    s3_connector_endpoint_url: str | None = Field(
        default=None,
        description="S3-compatible endpoint URL for S3Connector; unset means real AWS S3",
    )
    # botocore requires *a* region even for providers that ignore it
    # entirely; the connector falls back to us-east-1 when this is unset
    # rather than failing to construct a client.
    s3_connector_region: str | None = Field(
        default=None, description="AWS region name for S3Connector (defaults to us-east-1)"
    )
    s3_connector_bucket: str | None = Field(
        default=None,
        description="Default bucket for S3Connector reads given a bare key (no s3:// prefix)",
    )

    # The universal HTTP escape hatch: any JSON-over-HTTP API that has no
    # dedicated connector. One base URL per deployment, because a
    # connector is one *instance* of a configured system - a deployment
    # needing two REST APIs registers the second by constructing a second
    # `RESTConnector` directly, the same way `SQLConnector` is handled.
    rest_base_url: str | None = Field(
        default=None, description="Base URL of a JSON-over-HTTP API for RESTConnector"
    )
    # Split into name and value rather than one "Header: value" string so
    # nothing has to parse a colon out of a credential that may itself
    # contain colons (HTTP Basic's base64 of "user:pass" routinely does).
    rest_auth_header_name: str | None = Field(
        default=None,
        description="Auth header name for RESTConnector (defaults to Authorization)",
    )
    rest_auth_header_value: str | None = Field(
        default=None,
        description="Auth header value for RESTConnector, e.g. 'Bearer ...'",
    )

    graphql_url: str | None = Field(
        default=None, description="GraphQL endpoint URL for GraphQLConnector"
    )
    graphql_auth_header_name: str | None = Field(
        default=None,
        description="Auth header name for GraphQLConnector (defaults to Authorization)",
    )
    graphql_auth_header_value: str | None = Field(
        default=None,
        description="Auth header value for GraphQLConnector, e.g. 'Bearer ...'",
    )

    # Neo4j speaks Bolt, not HTTP - `neo4j://` (routing, for a cluster)
    # or `bolt://` (a single instance), optionally +s/+ssc for TLS.
    neo4j_uri: str | None = Field(
        default=None, description="Bolt URI for Neo4jConnector, e.g. neo4j://localhost:7687"
    )
    neo4j_user: str | None = Field(default=None, description="Username for Neo4jConnector")
    neo4j_password: str | None = Field(default=None, description="Password for Neo4jConnector")
    # Neo4j Community Edition has exactly one database ("neo4j"), which
    # the driver defaults to; this only matters on Enterprise/Aura, where
    # a deployment may keep several.
    neo4j_database: str | None = Field(
        default=None,
        description="Database name for Neo4jConnector; unset uses the server's default",
    )

    # Phase 11 (spec §16.2's "customer-db", §19's bootstrap philosophy).
    # DEMO-ONLY, deliberately narrow exception to the "no global DSN for
    # SQLConnector" principle documented on `praxis.connectors.sql.
    # connector.SQLConnector` itself ("connect to any SQL database" has
    # no single DSN to gate a factory on - registering one is left to
    # whoever needs it). This one field exists purely so Phase 11's own
    # seeded "customer-db" walkthrough (tests/e2e/
    # test_dashboard_walkthrough.py) has a single, real, named connector
    # `praxis.api.main` can register at startup - it is NOT a general
    # pattern for "the" external SQL database Praxis talks to, and nothing
    # else in this codebase should ever grow a second field like it.
    # Deliberately a SQLite DSN in practice (see
    # `praxis/demo/seed_customer_db.py`'s docstring for exactly why): any
    # *synthesized* skill's sandbox self-test can only import the Python
    # standard library, and stdlib has no Postgres driver but does have
    # `sqlite3` - so this is the one backend a synthesized "query this
    # DB" skill can actually be sandbox-validated against for real.
    demo_customer_db_dsn: str | None = Field(
        default=None,
        description=(
            "DEMO-ONLY: DSN for Phase 11's seeded 'customer-db' SQLConnector "
            "(spec §16.2's dashboard walkthrough); not a general pattern"
        ),
    )

    # Phase 9 (web-fetch safety tools, spec §6.1, §7). Genuinely optional
    # exactly like the three credentials above: unset means `WebConnector`
    # is simply not auto-registered into the bootstrap registry (spec §7:
    # "Enablement is a deployment-level toggle ... off by default, on by
    # config" - the one connector class that reaches the open internet,
    # unlike every other connector here which an operator already
    # explicitly wired up by registering it at all). No live Tavily key
    # is configured in this development environment - `WebConnector.search()`
    # raises a clear `SearchProviderNotConfiguredError` rather than a
    # silent empty-results no-op when this is unset (see
    # `praxis.connectors.web.web_connector`).
    tavily_api_key: str | None = Field(
        default=None, description="Tavily search API key for TavilySearchProvider"
    )

    # The explicit, off-by-default deployment toggle for the web tools
    # AS A GROUP (spec §7) - `web_search`/`web_read`/`web_crawl` self-
    # register (`praxis.agents.skills.web_search`/`web_read`/`web_crawl`)
    # only when this is set, regardless of whether `tavily_api_key` is
    # also set (a `web_read`/`web_crawl`-only deployment needs no Tavily
    # key at all, but still needs this switched on to make either tool
    # visible to the Planner). Declared here for documentation/
    # discoverability, but those three skill modules read the raw
    # `PRAXIS_WEB_TOOLS_ENABLED` env var directly at import time rather
    # than constructing a `Settings()` instance - constructing one would
    # make importing a skill module fail on a missing `database_url`
    # purely to decide whether to self-register, an unrelated hard
    # dependency no other skill module has.
    web_tools_enabled: bool = Field(
        default=False,
        description="Deployment-level toggle for web_search/web_read/web_crawl (off by default)",
    )

    # Universal MCP connectors (spec §6): each entry becomes one
    # MCPConnector, registered by praxis.connectors.bootstrap.build_registry
    # directly (not via the self-registering-factory mechanism the four
    # Phase 2 connectors use, since one entry here is one *instance*, not
    # one *type*). Setting, e.g.,
    #   PRAXIS_MCP_SERVERS='[{"name": "example", "command": "npx",
    #                         "args": ["-y", "some-mcp-server"]}]'
    # registers that MCP server as a connector with zero code changes -
    # the concrete proof that adding connector #N is a config entry, not
    # a new Python class. Defaults to empty: no MCP servers configured.
    # This is deliberately unlike the missing SQLConnector setting above:
    # "any SQL DB" has no single global DSN, but "which MCP servers this
    # deployment keeps connected" genuinely is a global, enumerable list
    # - each entry names one specific, already-known server - so it
    # belongs on Settings the way github_token/slack_bot_token/
    # prometheus_url do, not left to per-task construction.
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers to register as connectors; see MCPServerConfig",
    )

    # Which model serves which purpose. Empty means the catalogue's own
    # defaults; any key given here replaces one of them.
    #
    # This is what makes `praxis.llm.catalogue`'s "swapping providers is
    # a config change" literally true rather than aspirational: the
    # provider is derived from the model id, so pointing "planning" at
    # "gpt-5" moves planning to OpenAI with no code change and nothing
    # else to keep in sync. Overrides rather than a whole mapping, so a
    # deployment moving one purpose does not have to restate the others
    # and silently miss a new one added later.
    llm_model_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            'JSON object of LLM purpose to model id, e.g. {"planning": "gpt-5"}. '
            "Model ids beginning 'claude-' route to Anthropic, 'gpt-'/'o1'/'o3'/'o4' "
            "to OpenAI."
        ),
    )
