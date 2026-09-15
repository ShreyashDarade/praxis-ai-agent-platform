# praxis/core/interfaces.py
"""Core interface contracts every concrete backend implements (spec §4 DIP, §2)."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HealthStatus:
    name: str
    healthy: bool
    detail: str = ""


class HealthCheckable(abc.ABC):
    """Anything the platform can ask 'are you up?' (spec §6, §10, §18)."""

    @abc.abstractmethod
    async def health(self) -> HealthStatus: ...


@dataclass
class ConnectorDescription:
    """What describe() returns: the connector's shape/capabilities (spec §6)."""

    kind: str
    schema: dict[str, Any] = field(default_factory=dict)


class Connector(HealthCheckable):
    """Uniform contract for any external system Praxis talks to (spec §6)."""

    name: str
    read_only: bool = True

    @abc.abstractmethod
    async def describe(self) -> ConnectorDescription: ...

    @abc.abstractmethod
    async def read(self, query: str, **params: Any) -> Any: ...

    async def write(self, action: str, **params: Any) -> Any:
        if self.read_only:
            raise PermissionError(f"connector '{self.name}' is registered read-only")
        raise NotImplementedError


class RelationalStore(HealthCheckable):
    """Implemented in Task 4 by PostgresStore (spec §2)."""


class VectorStore(abc.ABC):
    @abc.abstractmethod
    async def upsert(self, doc_id: str, embedding: list[float], metadata: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    async def similarity_search(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]: ...


class GraphStore(abc.ABC):
    @abc.abstractmethod
    async def add_edge(
        self, source: str, relation: str, target: str, metadata: dict[str, Any] | None = None
    ) -> None: ...

    @abc.abstractmethod
    async def neighbors(self, node: str, relation: str | None = None) -> list[dict[str, Any]]: ...


class BlobStore(abc.ABC):
    @abc.abstractmethod
    async def put(self, key: str, data: bytes) -> str: ...

    @abc.abstractmethod
    async def get(self, key: str) -> bytes: ...


class Cache(abc.ABC):
    @abc.abstractmethod
    async def get(self, key: str) -> Any | None: ...

    @abc.abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...


class OcrEngine(abc.ABC):
    @abc.abstractmethod
    async def extract_text(self, image_bytes: bytes) -> str: ...


class Parser(abc.ABC):
    supported_mime_types: tuple[str, ...] = ()

    @abc.abstractmethod
    async def parse(self, data: bytes, mime_type: str) -> str: ...


class Chunker(abc.ABC):
    @abc.abstractmethod
    def chunk(self, text: str) -> list[str]: ...


class Embedder(abc.ABC):
    @abc.abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str


class SandboxExecutor(abc.ABC):
    @abc.abstractmethod
    async def run(self, code: str, *, timeout_seconds: int = 30) -> SandboxResult: ...
