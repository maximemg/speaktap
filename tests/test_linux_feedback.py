from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from speaktap.domain import CleanupStatus, TranscriptionResult
from speaktap.output.linux import (
    ClipboardOutput,
    TypingOutput,
    make_outputs,
    notify_status,
    play_sound,
)


def test_notification_is_not_a_transcript_delivering_output() -> None:
    """`notification` toggles state messages; it must never deliver the text.

    The service enables status notifications from this name but never routes
    the transcript through it, so make_outputs (text outputs only) does not
    know it. Anything that reintroduces a transcript-carrying notification,
    which would put dictated text in the notify-send argv, fails here.
    """

    with pytest.raises(ValueError, match="notification"):
        make_outputs(("notification",))


def test_make_outputs_builds_the_known_text_outputs() -> None:
    outputs = make_outputs(("typing", "clipboard"))

    assert tuple(output.output_id for output in outputs) == ("typing", "clipboard")


def _result(text: str) -> TranscriptionResult:
    return TranscriptionResult(
        session_id="session",
        raw_text=text,
        cleaned_text=None,
        cleanup_status=CleanupStatus.DISABLED,
        chunks=(),
    )


def _record_run(monkeypatch: Any) -> list[tuple[list[str], dict[str, Any]]]:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_typing_output_keeps_dictated_text_out_of_argv(monkeypatch: Any) -> None:
    """/proc/<pid>/cmdline is world readable, so argv must not carry the text."""

    calls = _record_run(monkeypatch)

    TypingOutput().deliver(_result("board meeting at nine"))

    command, kwargs = calls[0]
    assert "xdotool" in command
    assert all("board meeting" not in argument for argument in command)
    assert kwargs["input"] == b"board meeting at nine"


def test_clipboard_output_keeps_dictated_text_out_of_argv(monkeypatch: Any) -> None:
    calls = _record_run(monkeypatch)

    ClipboardOutput().deliver(_result("board meeting at nine"))

    command, kwargs = calls[0]
    assert "xclip" in command
    assert all("board meeting" not in argument for argument in command)
    assert kwargs["input"] == b"board meeting at nine"


def test_notification_contains_lifecycle_message(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command),
    )

    notify_status("Transcribing...")

    assert commands
    assert commands[0][-1] == "Transcribing..."
    assert "notify-send" in commands[0]


def test_sound_uses_paplay_without_blocking(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    sound = tmp_path / "toggle.oga"
    sound.touch()
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command),
    )

    play_sound(str(sound))

    assert commands == [["paplay", str(sound)]]


def test_disabled_sound_does_not_launch_player(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    sound = tmp_path / "toggle.oga"
    sound.touch()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("player must not launch")

    monkeypatch.setattr(subprocess, "Popen", fail)

    play_sound(str(sound), enabled=False)
