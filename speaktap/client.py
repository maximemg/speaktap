"""Tiny toggle/status client for the persistent SpeakTap service."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time

from .domain import ServiceState
from .protocol import (
    CancelCommand,
    Command,
    CommandResponse,
    ShutdownCommand,
    StartCommand,
    StatusCommand,
    StopCommand,
    decode_response,
    encode_command,
)
from .runtime import log_path, socket_path


def _send(command: Command, *, timeout: float = 360.0) -> CommandResponse:
    path = socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(encode_command(command))
        payload = bytearray()
        while b"\n" not in payload:
            chunk = client.recv(4096)
            if not chunk:
                break
            payload.extend(chunk)
    return decode_response(bytes(payload))


def _service_responds() -> bool:
    try:
        return _send(StatusCommand(), timeout=1.0).ok
    except ConnectionError, OSError, TimeoutError:
        return False


def _ensure_service() -> None:
    if _service_responds():
        return
    log = log_path().open("ab")
    process = subprocess.Popen(
        [sys.executable, "-m", "speaktap.service_app"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        close_fds=True,
    )
    deadline = time.monotonic() + 360.0
    while time.monotonic() < deadline:
        if _service_responds():
            log.close()
            return
        return_code = process.poll()
        if return_code is not None:
            log.close()
            raise RuntimeError(f"service exited with code {return_code}; inspect {log_path()}")
        time.sleep(0.25)
    log.close()
    raise RuntimeError(f"service did not become ready; inspect {log_path()}")


def _toggle(*, cleanup_enabled: bool | None) -> CommandResponse:
    status = _send(StatusCommand())
    if status.state is ServiceState.IDLE:
        return _send(StartCommand(cleanup_enabled=cleanup_enabled))
    if status.state is ServiceState.RECORDING:
        return _send(StopCommand())
    raise RuntimeError(f"service is busy in {status.state.value} state")


def main() -> None:
    parser = argparse.ArgumentParser(prog="speaktap")
    parser.add_argument(
        "command",
        choices=("toggle", "start", "stop", "cancel", "status", "service-stop"),
        default="toggle",
        nargs="?",
    )
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--cleanup", action="store_true", dest="cleanup_enabled")
    cleanup.add_argument("--no-cleanup", action="store_false", dest="cleanup_enabled")
    parser.set_defaults(cleanup_enabled=None)
    args = parser.parse_args()

    try:
        if args.command == "service-stop" and not _service_responds():
            print("service is not running")
            return
        _ensure_service()
        if args.command == "toggle":
            response = _toggle(
                cleanup_enabled=args.cleanup_enabled,
            )
        elif args.command == "start":
            response = _send(
                StartCommand(
                    cleanup_enabled=args.cleanup_enabled,
                )
            )
        elif args.command == "stop":
            response = _send(StopCommand())
        elif args.command == "cancel":
            response = _send(CancelCommand())
        elif args.command == "service-stop":
            response = _send(ShutdownCommand())
        else:
            response = _send(StatusCommand())
    except (ConnectionError, OSError, RuntimeError, TimeoutError) as error:
        parser.exit(1, f"dictation: {error}\n")

    if response.text:
        print(response.text)
    else:
        print(response.message)
    if not response.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
