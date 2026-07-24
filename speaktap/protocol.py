"""Language-neutral JSON-lines commands for the local SpeakTap service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from .domain import ServiceState


class ProtocolError(ValueError):
    """Raised when a client payload does not match the SpeakTap protocol."""


class CommandKind(StrEnum):
    START = "start"
    STOP = "stop"
    CANCEL = "cancel"
    STATUS = "status"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class StartCommand:
    kind: ClassVar[CommandKind] = CommandKind.START
    cleanup_enabled: bool | None = None


@dataclass(frozen=True, slots=True)
class StopCommand:
    kind: ClassVar[CommandKind] = CommandKind.STOP


@dataclass(frozen=True, slots=True)
class CancelCommand:
    kind: ClassVar[CommandKind] = CommandKind.CANCEL


@dataclass(frozen=True, slots=True)
class StatusCommand:
    kind: ClassVar[CommandKind] = CommandKind.STATUS


@dataclass(frozen=True, slots=True)
class ShutdownCommand:
    kind: ClassVar[CommandKind] = CommandKind.SHUTDOWN


type Command = StartCommand | StopCommand | CancelCommand | StatusCommand | ShutdownCommand


@dataclass(frozen=True, slots=True)
class CommandResponse:
    ok: bool
    state: ServiceState
    message: str
    session_id: str | None = None
    text: str | None = None


def encode_command(command: Command) -> bytes:
    body: dict[str, Any] = {"command": command.kind.value}
    if isinstance(command, StartCommand) and command.cleanup_enabled is not None:
        body["cleanup_enabled"] = command.cleanup_enabled
    return _encode(body)


def decode_command(payload: bytes) -> Command:
    body = _decode_object(payload)
    raw_kind = body.get("command")
    if not isinstance(raw_kind, str):
        raise ProtocolError("command must be start, stop, cancel, status, or shutdown")
    try:
        kind = CommandKind(raw_kind)
    except ValueError as error:
        raise ProtocolError("command must be start, stop, cancel, status, or shutdown") from error

    if kind is CommandKind.START:
        if "profile" in body:
            raise ProtocolError("ASR profile selection is install-time only")
        cleanup_enabled = body.get("cleanup_enabled")
        if cleanup_enabled is not None and not isinstance(cleanup_enabled, bool):
            raise ProtocolError("cleanup_enabled must be a boolean")
        return StartCommand(cleanup_enabled=cleanup_enabled)
    if kind is CommandKind.STOP:
        return StopCommand()
    if kind is CommandKind.CANCEL:
        return CancelCommand()
    if kind is CommandKind.SHUTDOWN:
        return ShutdownCommand()
    return StatusCommand()


def encode_response(response: CommandResponse) -> bytes:
    return _encode(
        {
            "ok": response.ok,
            "state": response.state.value,
            "message": response.message,
            "session_id": response.session_id,
            "text": response.text,
        }
    )


def decode_response(payload: bytes) -> CommandResponse:
    body = _decode_object(payload)
    ok = body.get("ok")
    message = body.get("message")
    session_id = body.get("session_id")
    text = body.get("text")
    if not isinstance(ok, bool):
        raise ProtocolError("ok must be a boolean")
    if not isinstance(message, str):
        raise ProtocolError("message must be a string")
    if session_id is not None and not isinstance(session_id, str):
        raise ProtocolError("session_id must be a string or null")
    if text is not None and not isinstance(text, str):
        raise ProtocolError("text must be a string or null")
    raw_state = body.get("state")
    if not isinstance(raw_state, str):
        raise ProtocolError("state is invalid")
    try:
        state = ServiceState(raw_state)
    except ValueError as error:
        raise ProtocolError("state is invalid") from error
    return CommandResponse(
        ok=ok,
        state=state,
        message=message,
        session_id=session_id,
        text=text,
    )


def _encode(body: dict[str, Any]) -> bytes:
    return (json.dumps(body, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _decode_object(payload: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("payload must be UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise ProtocolError("payload must be a JSON object")
    return decoded
