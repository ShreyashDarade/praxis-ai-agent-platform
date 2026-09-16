# praxis/observability/tracing.py
"""OpenTelemetry tracing (spec §10): "OpenTelemetry spans wrap
Orchestrator → subagent → tool execution → connector call; local
exporter for MVP."

`configure_tracing()` installs a process-wide `TracerProvider` with a
`ConsoleSpanExporter` - the "local exporter" spec §10 asks for.
`start_span(name, **attributes)` is a context manager wrapping one unit
of work as a span; real parent/child nesting falls out for free from
simply nesting these `with` blocks - OpenTelemetry's own context
propagation (not anything Praxis-specific) tracks "the current span" and
makes each new span a child of whatever span is active when it starts.
`praxis.core.orchestrator.Orchestrator` wraps a task's own execution and
each skill run; `praxis.agents.capability_factory.CapabilityFactory`
wraps a synthesis attempt and its connector `describe()` call - so a
real trace reads Orchestrator → skill-execution → connector-call, or
Orchestrator → capability-factory-synthesize → connector-call, matching
spec §10 exactly.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Span

_CONFIGURED = False
_SERVICE_NAME = "praxis"


def configure_tracing() -> None:
    """Installs the process-wide `TracerProvider` with a
    `ConsoleSpanExporter` behind a `SimpleSpanProcessor` - exports each
    span synchronously the moment it ends, deliberately not
    `BatchSpanProcessor`'s background-thread batching: this is a local,
    development-time exporter (spec §10's "local exporter for MVP"), not
    a high-throughput production sink, so there's nothing to gain from
    batching, and a background export thread racing process/test
    teardown (observed directly while building this: a stray "I/O
    operation on closed file" from a batch flush firing after a fast
    test process had already begun exiting) is a real cost with no
    corresponding benefit here.

    Idempotent - a second call is a no-op, matching
    `praxis.observability.logging.configure_logging`'s posture - so it's
    safe to call at `praxis.api.main` import time and again from a test.
    A test wanting a deterministic, isolated view of the span tree
    (rather than this module's own console output) attaches its own
    second `SimpleSpanProcessor` + in-memory exporter directly onto
    `opentelemetry.trace.get_tracer_provider()` after this has run -
    `TracerProvider` supports more than one processor at once, so the
    console exporter this installs keeps running alongside it.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_SERVICE_NAME)


@contextmanager
def start_span(name: str, **attributes: object) -> Iterator[Span]:
    """Starts a span named `name`, as a child of whichever span (if any)
    is currently active in this context - see the module docstring."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span
