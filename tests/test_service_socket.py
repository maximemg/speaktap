from __future__ import annotations

import socket
import time

import pytest

from speaktap.protocol import ProtocolError
from speaktap.service_app import _MAX_REQUEST_BYTES, _read_request


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
