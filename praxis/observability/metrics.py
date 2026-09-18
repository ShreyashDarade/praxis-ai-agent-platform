# praxis/observability/metrics.py
"""Real metrics: counters, histograms, and cost accounting (Prompt §9).

Prompt §9 requires *"latency, token, cost, cache-hit, error, and
success metrics"*. Praxis had structured log lines describing events,
which is not the same thing: you cannot compute a p95 or a cache-hit
rate from log prose without a log pipeline that parses it.

This module emits real OpenTelemetry instruments alongside the
existing logs, reusing the `opentelemetry-sdk` already in the
dependency set for tracing - no new dependency, and the same exporter
story.

**Why a local in-process registry as well as OTel instruments.** The
OTel metric reader in this deployment exports to the console, which
is fine for operations but useless for a *test* that needs to assert
"exactly one real API call happened" or "the cache was hit twice".
`MetricsRegistry` keeps the same numbers queryable in-process, which
is what makes the behaviour verifiable without standing up a metrics
backend. Both are updated from one call site so they cannot drift.

Cost is recorded here rather than in `praxis.agents.budget` on
purpose: the budget module *enforces* limits for one task tree, while
this module *observes* spend across the whole process. Conflating
them would mean an unbudgeted call went unmeasured.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import structlog

_logger = structlog.get_logger(__name__)

try:  # pragma: no cover - exercised implicitly; the fallback is the interesting path
    from opentelemetry import metrics as otel_metrics

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    _OTEL_AVAILABLE = False


@dataclass
class _Histogram:
    """Minimal streaming summary: enough for count/sum/min/max and
    quantiles over retained samples.

    Samples are capped so a long-running process cannot grow this
    without bound; beyond the cap, count/sum/min/max stay exact and
    only the quantiles become approximate over the retained window.
    That trade is stated rather than hidden, because a quantile that
    silently describes only the last N samples is otherwise
    misleading.
    """

    max_samples: int = 2048
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.samples) < self.max_samples:
            self.samples.append(value)

    def quantile(self, q: float) -> float | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": round(self.total, 6),
            "min": self.minimum,
            "max": self.maximum,
            "mean": round(self.total / self.count, 6) if self.count else None,
            "p50": self.quantile(0.50),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "samples_retained": len(self.samples),
            "quantiles_exact": self.count <= self.max_samples,
        }


class MetricsRegistry:
    """In-process counters and histograms, mirrored to OpenTelemetry.

    Thread-safe (unlike most of this codebase, which is
    single-threaded asyncio) because `asyncio.to_thread` is used for
    blocking work - sandbox runs, embedding, chart rendering - and
    those threads record durations here.
    """

    def __init__(self, *, service_name: str = "praxis") -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._otel_counters: dict[str, Any] = {}
        self._otel_histograms: dict[str, Any] = {}
        self._meter = (
            otel_metrics.get_meter(service_name) if _OTEL_AVAILABLE else None
        )

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{rendered}}}"

    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] = (
                self._counters.get(self._key(name, labels), 0.0) + value
            )
        if self._meter is not None:
            instrument = self._otel_counters.get(name)
            if instrument is None:
                instrument = self._meter.create_counter(name)
                self._otel_counters[name] = instrument
            instrument.add(value, labels or {})

    def observe(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = _Histogram()
                self._histograms[key] = histogram
            histogram.observe(value)
        if self._meter is not None:
            instrument = self._otel_histograms.get(name)
            if instrument is None:
                instrument = self._meter.create_histogram(name)
                self._otel_histograms[name] = instrument
            instrument.record(value, labels or {})

    @contextmanager
    def time(self, name: str, labels: dict[str, str] | None = None) -> Iterator[None]:
        """Records elapsed wall-clock into a histogram.

        Records on the failure path too - a latency metric that only
        counts successes systematically under-reports exactly the
        slow, failing calls an operator most needs to see.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started, labels)

    def counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self._counters.get(self._key(name, labels), 0.0)

    def histogram(self, name: str, labels: dict[str, str] | None = None) -> dict[str, Any]:
        with self._lock:
            histogram = self._histograms.get(self._key(name, labels))
        return histogram.to_dict() if histogram else {}

    def snapshot(self) -> dict[str, Any]:
        """Everything recorded so far - what `GET /metrics` serves."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    key: histogram.to_dict()
                    for key, histogram in self._histograms.items()
                },
            }

    def reset(self) -> None:
        """Clears all in-process state. Test-only."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Metric names, defined once so a producer and a consumer cannot
# disagree about spelling.
LLM_CALLS = "praxis.llm.calls"
LLM_TOKENS = "praxis.llm.tokens"
LLM_COST_USD = "praxis.llm.cost_usd"
LLM_LATENCY = "praxis.llm.latency_seconds"
CACHE_HITS = "praxis.cache.hits"
CACHE_MISSES = "praxis.cache.misses"
SKILL_EXECUTIONS = "praxis.skill.executions"
SKILL_LATENCY = "praxis.skill.latency_seconds"
SKILL_FAILURES = "praxis.skill.failures"
TASK_STARTED = "praxis.task.started"
TASK_COMPLETED = "praxis.task.completed"
TASK_FAILED = "praxis.task.failed"
TASK_LATENCY = "praxis.task.latency_seconds"
CONNECTOR_CALLS = "praxis.connector.calls"
CONNECTOR_FAILURES = "praxis.connector.failures"
DLQ_DEPTH = "praxis.dlq.depth"

# One shared registry. A module-level instance rather than a
# dependency-injected one because metrics are genuinely
# process-global: an operator asks "what is this process doing",
# not "what did this one object do".
metrics = MetricsRegistry()


def record_llm_call(
    *, model: str, purpose: str, tokens_in: int, tokens_out: int, cost_usd: float
) -> None:
    """Records one real model call across every relevant instrument."""
    labels = {"model": model, "purpose": purpose}
    metrics.increment(LLM_CALLS, labels=labels)
    metrics.increment(LLM_TOKENS, tokens_in + tokens_out, labels=labels)
    metrics.increment(LLM_COST_USD, cost_usd, labels=labels)


def record_cache(hit: bool, *, scope: str) -> None:
    metrics.increment(CACHE_HITS if hit else CACHE_MISSES, labels={"scope": scope})


def cache_hit_rate(scope: str) -> float | None:
    """The hit rate for one cache scope, or `None` when nothing has
    been recorded.

    `None` rather than 0.0 for "no data": a scope that has never been
    consulted has no hit rate, and reporting 0% would look like a
    performance problem that does not exist.
    """
    hits = metrics.counter(CACHE_HITS, {"scope": scope})
    misses = metrics.counter(CACHE_MISSES, {"scope": scope})
    total = hits + misses
    if total == 0:
        return None
    return hits / total
