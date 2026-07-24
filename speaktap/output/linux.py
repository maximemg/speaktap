"""Best-effort Linux desktop output adapters."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from ..domain import TranscriptionResult
from .base import OutputAdapter


class ClipboardOutput:
    output_id = "clipboard"

    def deliver(self, result: TranscriptionResult) -> None:
        text = result.output_text
        command = ["xclip", "-selection", "clipboard"]
        subprocess.run(
            command,
            input=text.encode(),
            check=True,
            timeout=5,
        )

    def close(self) -> None:
        return None


class TypingOutput:
    output_id = "typing"

    def deliver(self, result: TranscriptionResult) -> None:
        text = result.output_text
        if not text:
            return
        command = ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text]
        subprocess.run(command, check=True, timeout=15)

    def close(self) -> None:
        return None


class NotificationOutput:
    output_id = "notification"

    def deliver(self, result: TranscriptionResult) -> None:
        if not shutil.which("notify-send"):
            raise RuntimeError("notification output needs notify-send")
        summary = "Transcription"
        body = result.output_text or "No speech detected"
        subprocess.Popen(
            ["notify-send", summary, body[:500]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def close(self) -> None:
        return None


def make_outputs(names: tuple[str, ...]) -> tuple[OutputAdapter, ...]:
    factories: dict[str, Callable[[], OutputAdapter]] = {
        "clipboard": ClipboardOutput,
        "typing": TypingOutput,
        "notification": NotificationOutput,
    }
    unknown = sorted(set(names) - factories.keys())
    if unknown:
        raise ValueError(f"unsupported output adapter(s): {', '.join(unknown)}")
    return tuple(factories[name]() for name in names)


def deliver_outputs(
    outputs: tuple[OutputAdapter, ...],
    result: TranscriptionResult,
) -> tuple[str, ...]:
    errors: list[str] = []
    for output in outputs:
        try:
            output.deliver(result)
        except Exception as error:
            errors.append(f"{output.output_id}: {error}")
    return tuple(errors)


def notify_status(message: str, *, expire_milliseconds: int = 1_500) -> None:
    """Send transient state feedback without ever blocking the hotkey flow."""

    if not shutil.which("notify-send"):
        return
    subprocess.Popen(
        [
            "notify-send",
            "--expire-time",
            str(expire_milliseconds),
            "--hint",
            "int:transient:1",
            message,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def play_sound(
    path: str,
    *,
    enabled: bool = True,
    wait: bool = False,
) -> None:
    """Play a desktop cue through PulseAudio/PipeWire."""

    if not enabled or not Path(path).is_file() or not shutil.which("paplay"):
        return
    command = ["paplay", path]
    if wait:
        subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
