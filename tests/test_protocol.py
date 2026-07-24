from __future__ import annotations

import pytest

from speaktap.domain import ServiceState
from speaktap.protocol import (
    CancelCommand,
    CommandResponse,
    ProtocolError,
    ShutdownCommand,
    StartCommand,
    StatusCommand,
    StopCommand,
    decode_command,
    decode_response,
    encode_command,
    encode_response,
)


@pytest.mark.parametrize(
    "command",
    [
        StartCommand(cleanup_enabled=True),
        StopCommand(),
        CancelCommand(),
        StatusCommand(),
        ShutdownCommand(),
    ],
)
def test_command_round_trip(command: object) -> None:
    assert decode_command(encode_command(command)) == command  # type: ignore[arg-type]


def test_response_round_trip() -> None:
    response = CommandResponse(
        ok=True,
        state=ServiceState.RECORDING,
        message="started",
        session_id="session",
        text="hello world",
    )

    assert decode_response(encode_response(response)) == response


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        b'{"command":"unknown"}',
        b'{"command":"start","cleanup_enabled":"yes"}',
        b'{"command":"start","profile":"canary-1b-v2-int8"}',
    ],
)
def test_invalid_commands_rejected(payload: bytes) -> None:
    with pytest.raises(ProtocolError):
        decode_command(payload)
