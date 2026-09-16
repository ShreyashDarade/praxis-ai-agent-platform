# praxis/sandbox/executor.py
"""`DockerSandboxExecutor` (spec §8, §19, §20): runs arbitrary Python
code inside an ephemeral, network-isolated Docker container.

Standalone capability this phase, not yet wired into skill execution -
the hand-written skills registered this phase (`praxis.agents.skills`)
run in-process; Phase 6's Capability Factory is what will actually route
*synthesized* code through this executor before its first run (per spec
§8/§19/§20 - untrusted, LLM-generated code is exactly what a sandbox
protects against; a hand-written, reviewed skill is not).

Environment note (Windows + WSL2, no Docker Desktop on this machine):
`docker.from_env()` looks for a Windows named pipe or `DOCKER_HOST`;
neither exists here - the only real `dockerd` on this machine binds
`/var/run/docker.sock` *inside* a WSL2 distro, which Windows cannot dial
as a socket. Bridging that (a local TCP forwarder, or enabling `sshd`
inside WSL for an `ssh://` transport) requires opening a
network-reachable path to the Docker control API, which needs the
operator's explicit go-ahead - it was not enabled in this environment.
`docker_host`/`PRAXIS_DOCKER_HOST` exists precisely so a reachable
endpoint (a Docker Desktop npipe, a real remote host, or an
operator-approved bridge) can be plugged in via config with zero code
change, exactly like every other external dependency in this codebase.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout

from praxis.core.exceptions import SandboxViolationError
from praxis.core.interfaces import SandboxExecutor, SandboxResult

# A small, pinned base image - not "python:3.11" or "python:latest",
# which would silently drift in size/contents across pulls.
DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_MEM_LIMIT = "256m"


def build_docker_client(docker_host: str | None = None) -> docker.DockerClient:
    """Constructs a real `docker.DockerClient`.

    `docker_host` (typically `Settings().docker_host` /
    `PRAXIS_DOCKER_HOST`) takes precedence; falling back to
    `docker.from_env()` (which itself honors the `DOCKER_HOST` env var
    and platform defaults) when unset. Raises `DockerException` - never
    swallowed - if the daemon genuinely isn't reachable; callers (e.g.
    tests probing reachability) should catch that explicitly rather than
    this constructing a client that merely *looks* usable.
    """
    if docker_host:
        return docker.DockerClient(base_url=docker_host)
    return docker.from_env()


def is_docker_reachable(docker_host: str | None = None) -> bool:
    """A real connectivity probe (`client.ping()`), not an assumption.

    Used by this module's own tests to skip - with a clear reason,
    never a silent pass - the tests that need a genuinely reachable
    Docker daemon.
    """
    try:
        client = build_docker_client(docker_host)
        try:
            return bool(client.ping())
        finally:
            client.close()
    except DockerException:
        return False


class DockerSandboxExecutor(SandboxExecutor):
    """Runs Python code inside an ephemeral, ephemeral-per-call container.

    Every `run()` call: creates one detached container from a pinned
    base image, no network (`network_mode="none"`), a memory cap
    (`mem_limit`), waits up to `timeout_seconds` for it to finish,
    captures stdout/stderr, and always removes the container afterward
    - even on failure or timeout, so nothing is ever left behind.
    """

    def __init__(
        self,
        *,
        docker_host: str | None = None,
        image: str = DEFAULT_IMAGE,
        mem_limit: str = DEFAULT_MEM_LIMIT,
    ) -> None:
        self._image = image
        self._mem_limit = mem_limit
        self._client = build_docker_client(docker_host)
        self._ensure_image()

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(self._image)
        except ImageNotFound:
            self._client.images.pull(self._image)

    async def run(self, code: str, *, timeout_seconds: int = 30) -> SandboxResult:
        # docker-py is a synchronous/blocking library (it makes plain
        # `requests` calls under the hood) - run it off the event loop
        # thread so a slow/hanging Docker call never blocks the whole
        # asyncio loop this executor is called from.
        return await asyncio.to_thread(self._run_blocking, code, timeout_seconds)

    def _run_blocking(self, code: str, timeout_seconds: int) -> SandboxResult:
        container_name = f"praxis-sandbox-{uuid.uuid4().hex[:12]}"
        container = self._client.containers.run(
            self._image,
            ["python", "-c", code],
            name=container_name,
            detach=True,
            network_mode="none",
            mem_limit=self._mem_limit,
            stdout=True,
            stderr=True,
        )
        try:
            try:
                wait_result: Any = container.wait(timeout=timeout_seconds)
                exit_code = (
                    wait_result.get("StatusCode", -1)
                    if isinstance(wait_result, dict)
                    else int(wait_result)
                )
                stdout = _decode(container.logs(stdout=True, stderr=False))
                stderr = _decode(container.logs(stdout=False, stderr=True))

                if _was_oom_killed(container):
                    # A genuine resource-bound violation (spec §12) - not
                    # "the user's code returned non-zero" (an ordinary
                    # failure, still a plain SandboxResult below). Docker
                    # itself reports this via the container's own real
                    # inspect result (`State.OOMKilled`), never guessed
                    # from the exit code alone (137 is also just "some
                    # process was SIGKILLed", e.g. our own timeout-kill
                    # branch below) - verified directly against a real
                    # container that actually over-allocated past its
                    # `mem_limit` while building this.
                    raise SandboxViolationError(
                        f"sandboxed execution in container '{container_name}' exceeded its "
                        f"memory bound (mem_limit={self._mem_limit}) and was OOM-killed",
                        violation="oom_killed",
                        exit_code=exit_code,
                        detail=(
                            f"Docker reported State.OOMKilled=true, exit_code={exit_code}; "
                            f"stdout:\n{stdout}\nstderr:\n{stderr}"
                        ),
                    )
                return SandboxResult(exit_code=exit_code, stdout=stdout, stderr=stderr)
            except (ReadTimeout, RequestsConnectionError):
                # The container genuinely did not finish within
                # timeout_seconds (e.g. it's sleeping/looping) - kill it
                # rather than let it run forever, and report a clear,
                # non-zero, non-hanging timeout outcome (spec §12: a
                # timeout is never silently represented as success).
                stdout = _try_logs(container, stdout=True, stderr=False)
                try:
                    container.kill()
                except (NotFound, DockerException):
                    pass
                stderr = f"sandbox execution timed out after {timeout_seconds}s and was killed"
                return SandboxResult(exit_code=124, stdout=stdout, stderr=stderr)
        finally:
            try:
                container.remove(force=True)
            except (NotFound, DockerException):
                pass


def _was_oom_killed(container: Any) -> bool:
    """Real detection, not a guess: refreshes the container's own state
    (`container.reload()`) and reads Docker's own `State.OOMKilled`
    field - the authoritative signal the daemon itself sets the moment
    the kernel's cgroup OOM killer terminates a container for exceeding
    its `mem_limit`. Deliberately does not treat exit code 137 alone as
    sufficient (that's just "some process got SIGKILLed" - e.g. this
    same module's own timeout-triggered `container.kill()` below also
    produces 137, and is *not* a resource violation).
    """
    try:
        container.reload()
    except DockerException:
        return False
    return bool(container.attrs.get("State", {}).get("OOMKilled", False))


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _try_logs(container: Any, *, stdout: bool, stderr: bool) -> str:
    try:
        return _decode(container.logs(stdout=stdout, stderr=stderr))
    except DockerException:
        return ""
