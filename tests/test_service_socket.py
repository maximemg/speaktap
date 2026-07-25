from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from speaktap.domain import ServiceState
from speaktap.protocol import (
    Command,
    CommandResponse,
    ProtocolError,
    ShutdownCommand,
    decode_response,
)
from speaktap.service import ServiceSnapshot
from speaktap.service_app import _MAX_REQUEST_BYTES, _read_request, _serve


class StubService:
    """Expose only the surface the socket loop is allowed to depend on.

    It deliberately has no _state attribute, so any reach into the service
    internals fails here instead of silently coupling transport to state.
    """

    def __init__(self) -> None:
        self.shutdown_requested = threading.Event()
        self.commands: list[Command] = []

    def snapshot(self) -> ServiceSnapshot:
        return ServiceSnapshot(
            state=ServiceState.IDLE,
            session_id=None,
            options=None,
            last_error=None,
        )

    def handle(self, command: Command) -> CommandResponse:
        self.commands.append(command)
        if isinstance(command, ShutdownCommand):
            self.shutdown_requested.set()
        return CommandResponse(ok=True, state=ServiceState.IDLE, message="handled")


def _request(path: Path, payload: bytes) -> CommandResponse:
    deadline = time.monotonic() + 5.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5.0)
        client.connect(str(path))
        client.sendall(payload)
        received = bytearray()
        while b"\n" not in received:
            chunk = client.recv(4096)
            if not chunk:
                break
            received.extend(chunk)
    return decode_response(bytes(received))


def test_read_request_accepts_a_newline_terminated_payload() -> None:
    server, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with server, client:
        client.sendall(b'{"command":"status"}\n')

        assert _read_request(server) == b'{"command":"status"}\n'


def test_read_request_rejects_an_oversized_payload() -> None:
    server, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with server, client:
        client.sendall(b"x" * (_MAX_REQUEST_BYTES + 1024))

        with pytest.raises(ProtocolError, match="too large"):
            _read_request(server)


def test_read_request_gives_up_on_a_client_that_never_sends() -> None:
    """A silent peer must not wedge the single-threaded accept loop.

    Accepted sockets are blocking regardless of the listening socket timeout,
    so without an explicit read timeout the service hangs here forever and
    stops responding to SIGTERM as well.
    """

    server, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with server, client:
        started = time.monotonic()

        with pytest.raises(TimeoutError):
            _read_request(server, timeout_seconds=0.2)

        assert time.monotonic() - started < 5.0


def _run_serve(service: StubService, path: Path) -> tuple[threading.Thread, threading.Event]:
    stopping = threading.Event()
    thread = threading.Thread(
        target=_serve,
        args=(service, path, stopping),
        daemon=True,
    )
    thread.start()
    return thread, stopping


def test_serve_round_trips_a_command(tmp_path: Path) -> None:
    service = StubService()
    path = tmp_path / "service.sock"
    thread, stopping = _run_serve(service, path)
    try:
        response = _request(path, b'{"command":"status"}\n')
    finally:
        stopping.set()
        thread.join(timeout=5.0)

    assert response.ok
    assert response.message == "handled"
    assert not thread.is_alive()


def test_serve_answers_a_malformed_payload_from_the_public_surface(
    tmp_path: Path,
) -> None:
    """The error path must build its reply without reaching into the service."""

    service = StubService()
    path = tmp_path / "service.sock"
    thread, stopping = _run_serve(service, path)
    try:
        response = _request(path, b"this is not json\n")
    finally:
        stopping.set()
        thread.join(timeout=5.0)

    assert not response.ok
    assert response.state is ServiceState.IDLE
    assert service.commands == []
