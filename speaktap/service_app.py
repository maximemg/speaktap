"""Persistent local SpeakTap service."""

from __future__ import annotations

import fcntl
import signal
import socket
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import FrameType

from .capture import ArecordSource
from .config import SpeakTapConfig
from .diagnostics import SessionLogger
from .domain import ServiceState, SessionOptions
from .factory import make_pipeline
from .output import deliver_outputs, make_outputs, notify_status, play_sound
from .pipeline import PipelineSession
from .protocol import (
    CancelCommand,
    Command,
    CommandResponse,
    ProtocolError,
    ShutdownCommand,
    StartCommand,
    StatusCommand,
    StopCommand,
    decode_command,
    encode_response,
)
from .runtime import lock_path, session_log_path, socket_path
from .service import InvalidTransitionError, ServiceStateMachine
from .transcription import OnnxAsrBackend

_MAX_REQUEST_BYTES = 65_536


class AsrService:
    def __init__(self, config: SpeakTapConfig) -> None:
        self._config = config
        self._state = ServiceStateMachine()
        self._backend = OnnxAsrBackend(
            profile=config.model_profile,
            provider=config.execution_provider,
            threads=config.asr_threads,
            inter_op_threads=config.asr_inter_op_threads,
            execution_mode=config.asr_execution_mode,
        )
        self._pipeline: PipelineSession | None = None
        self._notifications_enabled = "notification" in config.outputs
        self._outputs = make_outputs(
            tuple(name for name in config.outputs if name != "notification")
        )
        self._session_logger = SessionLogger(session_log_path())
        self.shutdown_requested = threading.Event()

    def prepare(self) -> None:
        self._backend.load()
        self._backend.warmup()

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.cancel()
        for output in self._outputs:
            output.close()
        self._backend.close()

    def handle(self, command: Command) -> CommandResponse:
        try:
            if isinstance(command, StatusCommand):
                snapshot = self._state.snapshot()
                message = snapshot.last_error or snapshot.state.value
                return CommandResponse(
                    ok=True,
                    state=snapshot.state,
                    message=message,
                    session_id=snapshot.session_id,
                )
            if isinstance(command, StartCommand):
                return self._start(command)
            if isinstance(command, StopCommand):
                return self._stop()
            if isinstance(command, CancelCommand):
                return self._cancel()
            if isinstance(command, ShutdownCommand):
                return self._shutdown()
            raise TypeError("unsupported command")
        except (InvalidTransitionError, ValueError, RuntimeError) as error:
            snapshot = self._state.snapshot()
            return CommandResponse(
                ok=False,
                state=snapshot.state,
                message=str(error),
                session_id=snapshot.session_id,
            )

    def _start(self, command: StartCommand) -> CommandResponse:
        cleanup_enabled = (
            self._config.cleanup_enabled
            if command.cleanup_enabled is None
            else command.cleanup_enabled
        )
        snapshot = self._state.start(
            SessionOptions(
                asr_profile=self._config.asr_profile,
                cleanup_enabled=cleanup_enabled,
            )
        )
        if snapshot.session_id is None:
            raise RuntimeError("state machine did not create a session")
        pipeline = make_pipeline(self._config, self._backend)
        try:
            play_sound(
                self._config.start_sound,
                enabled=self._config.sounds_enabled,
                wait=True,
            )
            pipeline.start(
                ArecordSource(
                    device=self._config.audio_device,
                    sample_rate=self._config.sample_rate,
                    frame_milliseconds=self._config.frame_milliseconds,
                ),
                session_id=snapshot.session_id,
                cleanup_enabled=cleanup_enabled,
            )
        except Exception as error:
            self._state.fail(error)
            raise
        self._pipeline = pipeline
        self._notify(f"Recording started (maximum {self._config.max_recording_seconds:g}s)…")
        return CommandResponse(
            ok=True,
            state=ServiceState.RECORDING,
            message="recording started",
            session_id=snapshot.session_id,
        )

    def _stop(self) -> CommandResponse:
        started = time.monotonic()
        snapshot = self._state.begin_finalization()
        self._notify("Transcribing…")
        pipeline = self._pipeline
        if pipeline is None:
            self._state.fail("recording pipeline is missing")
            raise RuntimeError("recording pipeline is missing")
        try:
            result = pipeline.stop(
                on_capture_stopped=lambda: play_sound(
                    self._config.stop_sound,
                    enabled=self._config.sounds_enabled,
                )
            )
            self._state.begin_output()
            output_errors = deliver_outputs(self._outputs, result)
            elapsed_ms = round((time.monotonic() - started) * 1000)
            try:
                self._session_logger.success(
                    result,
                    service_post_stop_ms=elapsed_ms,
                    output_errors=output_errors,
                )
            except OSError as log_error:
                output_errors += (f"diagnostics: {log_error}",)
            self._notify(self._completion_message(result.output_text, output_errors, elapsed_ms))
            self._state.complete()
            self._pipeline = None
        except Exception as error:
            pipeline.cancel()
            self._pipeline = None
            self._state.fail(error)
            elapsed_ms = round((time.monotonic() - started) * 1000)
            with suppress(OSError):
                self._session_logger.failure(
                    session_id=snapshot.session_id,
                    error=error,
                    service_post_stop_ms=elapsed_ms,
                )
            self._notify(f"Transcription failed ({elapsed_ms}ms).")
            raise
        message = "transcription complete"
        if output_errors:
            message += f" (output warnings: {'; '.join(output_errors)})"
        return CommandResponse(
            ok=True,
            state=ServiceState.IDLE,
            message=message,
            session_id=snapshot.session_id,
            text=result.output_text,
        )

    def _cancel(self) -> CommandResponse:
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.cancel()
            self._pipeline = None
        self._state.cancel()
        play_sound(
            self._config.stop_sound,
            enabled=self._config.sounds_enabled,
        )
        self._notify("Recording cancelled")
        return CommandResponse(
            ok=True,
            state=ServiceState.IDLE,
            message="recording cancelled",
        )

    def _shutdown(self) -> CommandResponse:
        if self._pipeline is not None:
            self._pipeline.cancel()
            self._pipeline = None
            self._state.cancel()
        self.shutdown_requested.set()
        return CommandResponse(
            ok=True,
            state=ServiceState.IDLE,
            message="service stopped",
        )

    def _notify(self, message: str) -> None:
        if self._notifications_enabled:
            notify_status(message)

    def _completion_message(
        self,
        text: str,
        output_errors: tuple[str, ...],
        elapsed_ms: int,
    ) -> str:
        if not text:
            return f"Transcription empty ({elapsed_ms}ms)."
        failed = {error.partition(":")[0] for error in output_errors}
        configured = {output.output_id for output in self._outputs}
        if "typing" in configured and "typing" not in failed:
            return f"Transcribed and typed ({elapsed_ms}ms)."
        if "clipboard" in configured and "clipboard" not in failed:
            return f"Transcription copied to clipboard ({elapsed_ms}ms)."
        return f"Transcription complete ({elapsed_ms}ms)."


def _read_request(connection: socket.socket) -> bytes:
    payload = bytearray()
    while len(payload) <= _MAX_REQUEST_BYTES:
        chunk = connection.recv(4096)
        if not chunk:
            break
        payload.extend(chunk)
        if b"\n" in chunk:
            break
    if len(payload) > _MAX_REQUEST_BYTES:
        raise ProtocolError("request is too large")
    return bytes(payload)


def _serve(service: AsrService, path: Path, stopping: threading.Event) -> None:
    if path.exists():
        path.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        path.chmod(0o600)
        server.listen(8)
        server.settimeout(1.0)
        while not stopping.is_set() and not service.shutdown_requested.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                try:
                    response = service.handle(decode_command(_read_request(connection)))
                except Exception as error:
                    snapshot = service._state.snapshot()
                    response = CommandResponse(
                        ok=False,
                        state=snapshot.state,
                        message=str(error),
                        session_id=snapshot.session_id,
                    )
                with suppress(BrokenPipeError, ConnectionResetError):
                    connection.sendall(encode_response(response))
    path.unlink(missing_ok=True)


def main() -> None:
    lock_file = lock_path().open("a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Dictation service is already running", file=sys.stderr)
        raise SystemExit(2) from None

    stopping = threading.Event()

    def stop_service(_signum: int, _frame: FrameType | None) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop_service)
    signal.signal(signal.SIGTERM, stop_service)
    service = AsrService(SpeakTapConfig.load())
    path = socket_path()
    try:
        service.prepare()
        _serve(service, path, stopping)
    finally:
        path.unlink(missing_ok=True)
        service.close()
        lock_file.close()


if __name__ == "__main__":
    main()
