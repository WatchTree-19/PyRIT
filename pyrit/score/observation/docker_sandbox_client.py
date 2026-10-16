# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run programs in disposable, locked-down Docker containers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pyrit.models import ComponentIdentifier, ProgramExecutionEvent
from pyrit.score.observation.sandbox_client import SandboxExecutionError, SandboxRun

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.models import ProgramInvocation

logger = logging.getLogger(__name__)

_DIGEST_PATTERN = re.compile(r"@(sha256:[0-9a-f]{64})$")
_NEVER_STARTED = "0001-01-01T00:00:00Z"


@dataclass(frozen=True)
class DockerRuntimeProfile:
    """
    One allowlisted runtime: a digest-pinned image and the command that runs a program in it.

    The container runs ``command``, then the program's path, then the invocation's arguments.
    For example ``command=("python3", "-I")`` runs ``python3 -I /program/main.py <args>``.
    ``image`` must be pinned by digest, so the profile always names the same runtime.
    """

    image: str
    command: tuple[str, ...]
    program_filename: str = "main.py"

    def __post_init__(self) -> None:
        """
        Validate the image pin and the program file name.

        Raises:
            ValueError: If the image is not pinned by digest, the command is empty, or the
                file name is not a plain name.
        """
        if _DIGEST_PATTERN.search(self.image) is None:
            raise ValueError(f"Runtime image {self.image!r} must be pinned by digest (name@sha256:<64 hex>).")
        if not self.command or any(not part for part in self.command):
            raise ValueError("A runtime profile command must be a nonempty argument list.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.program_filename):
            raise ValueError(f"Program file name {self.program_filename!r} must be a plain file name.")

    @property
    def image_digest(self) -> str:
        """The ``sha256:`` digest the image is pinned to; ``__post_init__`` guarantees its presence."""
        return self.image.rsplit("@", 1)[1]


@dataclass(frozen=True)
class _ContainerExit:
    exit_code: int | None
    oom_killed: bool


class DockerSandboxClient:
    """
    Run each invocation in its own disposable Docker container.

    Every container has no network, a read-only root file system, all capabilities dropped,
    no privilege escalation, an unprivileged user, and caps on memory, CPU, and process count.
    The program is mounted read-only; a small ``/tmp`` is the only writable location. Each
    invocation gets a fresh container that is removed afterwards, so one run cannot leave state
    for the next.

    Images are never pulled while scoring: a profile's image must already be present, or the
    attempt fails as a sandbox error. Pull the pinned images before a run.

    A container that runs past ``timeout_seconds`` is killed and reported as timed out. One that
    the kernel stops for exceeding its memory limit is reported as exceeding a resource limit.
    Each stream keeps at most ``max_output_bytes`` bytes. The timeout includes container start-up
    and the attached client's exit; a client still running once its streams have closed is
    ended at the deadline, and the container's own state then decides the result.

    This client isolates submitted code with Docker's defaults plus the restrictions above. It
    is not a hardened multi-tenant sandbox; run it on a host dedicated to evaluation.
    """

    _READ_CHUNK_BYTES = 1 << 16
    _CONTROL_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        *,
        profiles: Mapping[str, DockerRuntimeProfile],
        docker_executable: str = "docker",
        timeout_seconds: float = 10.0,
        memory: str = "256m",
        cpus: float = 1.0,
        pids_limit: int = 64,
        max_output_bytes: int = 64_000,
    ) -> None:
        """
        Initialize with the allowlisted runtime profiles and per-container limits.

        Args:
            profiles (Mapping[str, DockerRuntimeProfile]): Runtime profile names and their runtimes.
            docker_executable (str): The Docker command line client to call.
            timeout_seconds (float): Wall-clock limit for each invocation.
            memory (str): Memory limit for each container, in Docker's notation, such as ``256m``.
            cpus (float): CPU limit for each container.
            pids_limit (int): Most processes a container may hold.
            max_output_bytes (int): Most bytes of each stream kept per invocation.

        Raises:
            ValueError: If no profile is given, a profile name is invalid, or a limit is not positive.
        """
        if not profiles:
            raise ValueError("DockerSandboxClient requires at least one runtime profile.")
        invalid = [name for name in profiles if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name)]
        if invalid:
            raise ValueError(f"Invalid runtime profile names: {invalid}.")
        if timeout_seconds <= 0 or cpus <= 0 or pids_limit < 1 or max_output_bytes < 1:
            raise ValueError("timeout_seconds, cpus, pids_limit and max_output_bytes must be positive.")
        if not re.fullmatch(r"[1-9][0-9]*[bkmg]?", memory):
            raise ValueError(f"Memory limit {memory!r} must look like 256m or 1g.")
        self._profiles = dict(profiles)
        self._docker = docker_executable
        self._timeout_seconds = timeout_seconds
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._max_output_bytes = max_output_bytes

    def get_identifier(self) -> ComponentIdentifier:
        """
        Return the profiles and limits.

        Returns:
            ComponentIdentifier: The nonsecret client configuration.
        """
        return ComponentIdentifier.of(
            self,
            params={
                "client_version": 1,
                "profiles": {
                    name: {
                        "image": profile.image,
                        "command": list(profile.command),
                        "program_filename": profile.program_filename,
                    }
                    for name, profile in sorted(self._profiles.items())
                },
                "timeout_seconds": self._timeout_seconds,
                "memory": self._memory,
                "cpus": self._cpus,
                "pids_limit": self._pids_limit,
                "max_output_bytes": self._max_output_bytes,
            },
        )

    def get_runtime_profiles(self) -> frozenset[str]:
        """
        Return the configured runtime profile names.

        Returns:
            frozenset[str]: The profile names this client can run.
        """
        return frozenset(self._profiles)

    async def execute_async(
        self, *, program: str, runtime_profile: str, invocations: tuple[ProgramInvocation, ...]
    ) -> SandboxRun:
        """
        Run each invocation in a fresh container, in order.

        Args:
            program (str): The program text, written to a read-only file in the container.
            runtime_profile (str): The allowlisted profile to run under.
            invocations (tuple[ProgramInvocation, ...]): The arguments and input of each run.

        Returns:
            SandboxRun: One event per invocation that ran. If the sandbox fails after some
            invocations ran, the run holds those events and the reason it stopped.

        Raises:
            ValueError: If the runtime profile is not configured.
            SandboxExecutionError: If no invocation could be run.
        """
        profile = self._profiles.get(runtime_profile)
        if profile is None:
            raise ValueError(f"Runtime profile {runtime_profile!r} is not configured.")
        program_dir = Path(tempfile.mkdtemp(prefix="pyrit-program-"))
        program_path = program_dir / profile.program_filename
        try:
            program_path.write_text(program, encoding="utf-8")
            program_path.chmod(0o444)
            program_dir.chmod(0o755)
            events: list[ProgramExecutionEvent] = []
            for index, invocation in enumerate(invocations):
                try:
                    event, cleanup_warning = await self._run_invocation_async(
                        case_index=index,
                        runtime_profile=runtime_profile,
                        profile=profile,
                        program_dir=program_dir,
                        invocation=invocation,
                    )
                except SandboxExecutionError as error:
                    if not events:
                        raise
                    return SandboxRun(events=tuple(events), reasons=(f"sandbox error at case {index}: {error}",))
                events.append(event)
                if cleanup_warning is not None:
                    # The environment may still be running, so no further case starts beside it.
                    remaining = len(invocations) - len(events)
                    reasons = (f"stopped after case {index}: {cleanup_warning}",) if remaining else ()
                    return SandboxRun(events=tuple(events), reasons=reasons, warnings=(cleanup_warning,))
            return SandboxRun(events=tuple(events))
        finally:
            self._remove_program_dir(program_dir=program_dir, program_path=program_path)

    @staticmethod
    def _remove_program_dir(*, program_dir: Path, program_path: Path) -> None:
        """
        Delete the submitted program, restoring write permission first.

        On Windows the read-only mode is a file attribute that blocks deletion, so the file
        is made writable again before the directory is removed.
        """
        with contextlib.suppress(OSError):
            program_path.chmod(stat.S_IREAD | stat.S_IWRITE)
        shutil.rmtree(program_dir, ignore_errors=True)
        if program_dir.exists():
            logger.warning("Could not remove the program directory %s", program_dir)

    async def _run_invocation_async(
        self,
        *,
        case_index: int,
        runtime_profile: str,
        profile: DockerRuntimeProfile,
        program_dir: Path,
        invocation: ProgramInvocation,
    ) -> tuple[ProgramExecutionEvent, str | None]:
        """
        Run one invocation in a new container and remove the container afterwards.

        Returns:
            tuple[ProgramExecutionEvent, str | None]: What the invocation did, and a warning when
            the container could not be removed.

        Raises:
            SandboxExecutionError: If Docker cannot start or inspect the container, or the
                container had not finished when the attached client exited.
        """
        name = f"pyrit-sandbox-{uuid.uuid4().hex}"
        resolved_command = (*profile.command, f"/program/{profile.program_filename}", *invocation.arguments)
        argv = [
            self._docker,
            "run",
            "--name",
            name,
            "--interactive",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--memory",
            self._memory,
            "--memory-swap",
            self._memory,
            "--cpus",
            str(self._cpus),
            "--pids-limit",
            str(self._pids_limit),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=16m",
            "--workdir",
            "/tmp",
            "--mount",
            f"type=bind,source={os.fspath(program_dir)},target=/program,readonly",
            profile.image,
            *resolved_command,
        ]
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise SandboxExecutionError(f"could not start {self._docker!r}: {error}") from error
        timed_out = False
        cleanup_warning: str | None = None
        # One wall-clock deadline covers the whole invocation: reading the output and the
        # attached client's exit.
        deadline = started + self._timeout_seconds
        try:
            readers = asyncio.gather(
                self._read_capped_async(process.stdout),
                self._read_capped_async(process.stderr),
                self._feed_stdin_async(process.stdin, invocation.stdin),
            )
            try:
                (stdout, stdout_truncated), (stderr, stderr_truncated), _ = await asyncio.wait_for(
                    asyncio.shield(readers), timeout=self._timeout_seconds
                )
            except TimeoutError:
                timed_out = True
                await self._control_async("kill", name)
                try:
                    await asyncio.wait_for(asyncio.shield(readers), timeout=self._CONTROL_TIMEOUT_SECONDS)
                except TimeoutError:
                    # The container would not stop; ending the attached client closes its streams.
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                (stdout, stdout_truncated), (stderr, stderr_truncated), _ = await readers
            try:
                exit_budget = self._CONTROL_TIMEOUT_SECONDS if timed_out else max(deadline - time.monotonic(), 0.0)
                await asyncio.wait_for(process.wait(), timeout=exit_budget)
            except TimeoutError:
                # The streams closed but the attached client did not exit in time. Ending the
                # client keeps the invocation bounded; the container's own state, read below,
                # decides whether the program finished, so a stalled client is not mistaken
                # for a program that ran too long.
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            duration_ms = int((time.monotonic() - started) * 1000)
            container = await self._inspect_async(name=name, cli_stderr=stderr, timed_out=timed_out)
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            removal_status, removal_output = await self._control_async("rm", "--force", name)
            if removal_status != 0:
                detail = removal_output.strip()[:300] or "unknown Docker failure"
                cleanup_warning = f"could not remove container {name}: {detail}"
                logger.warning("Sandbox cleanup failed: %s", cleanup_warning)
        event = ProgramExecutionEvent(
            case_index=case_index,
            runtime_profile=runtime_profile,
            runtime_image_digest=profile.image_digest,
            resolved_command=resolved_command,
            exit_code=None if (timed_out or container.oom_killed) else container.exit_code,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            resource_limit_exceeded=container.oom_killed and not timed_out,
            duration_ms=duration_ms,
        )
        return event, cleanup_warning

    async def _inspect_async(self, *, name: str, cli_stderr: bytes, timed_out: bool) -> _ContainerExit:
        """
        Read how the container ended, distinguishing a program exit from a Docker failure.

        An exit code is only reported for a container that has finished. A container still
        running when the attached client exited means the run was cut short by Docker, not by
        the program, unless the sandbox stopped it at its time limit.

        Returns:
            _ContainerExit: The container's exit code and whether the kernel killed it for memory.

        Raises:
            SandboxExecutionError: If the container does not exist, never started, or had not
                finished when the client exited.
        """
        returncode, output = await self._control_async(
            "inspect",
            "--format",
            "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.State.StartedAt}}",
            name,
        )
        fields = output.split()
        if returncode != 0 or len(fields) != 4 or fields[3] == _NEVER_STARTED:
            lines = [line.strip() for line in cli_stderr.decode("utf-8", errors="replace").splitlines() if line.strip()]
            detail = lines[0][:500] if lines else output.strip()[:500]
            raise SandboxExecutionError(f"the container did not start: {detail or 'unknown Docker failure'}")
        status = fields[0]
        if status != "exited" and not timed_out:
            raise SandboxExecutionError(
                f"the Docker client exited while the container was still {status}, so the program did not finish"
            )
        return _ContainerExit(exit_code=int(fields[1]), oom_killed=fields[2] == "true")

    async def _control_async(self, *arguments: str) -> tuple[int, str]:
        """
        Run a short Docker management command, bounded in time.

        Returns:
            tuple[int, str]: The exit status and combined output; status -1 when it failed to run.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self._docker,
                *arguments,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as error:
            logger.warning("Docker %s failed to start: %s", arguments[0], error)
            return -1, str(error)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=self._CONTROL_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("Docker %s timed out", arguments[0])
            return -1, "timed out"
        finally:
            # Runs on timeout and on cancellation alike, so no management command outlives its caller.
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        return process.returncode if process.returncode is not None else -1, output.decode("utf-8", errors="replace")

    async def _read_capped_async(self, stream: asyncio.StreamReader | None) -> tuple[bytes, bool]:
        """
        Read a stream to its end, keeping at most the configured number of bytes.

        Returns:
            tuple[bytes, bool]: The retained bytes and whether more were discarded.
        """
        if stream is None:
            return b"", False
        kept = bytearray()
        truncated = False
        while chunk := await stream.read(self._READ_CHUNK_BYTES):
            room = self._max_output_bytes - len(kept)
            if len(chunk) > room:
                truncated = True
            kept.extend(chunk[: max(room, 0)])
        return bytes(kept), truncated

    @staticmethod
    async def _feed_stdin_async(stream: asyncio.StreamWriter | None, data: str | None) -> None:
        """Write the invocation's input, then close standard input so the program sees its end."""
        if stream is None:
            return
        try:
            if data:
                stream.write(data.encode("utf-8"))
                await stream.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await stream.wait_closed()
