from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from speaktap.output.linux import notify_status, play_sound


def test_notification_contains_lifecycle_message(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command),
    )

    notify_status("Transcribing…")

    assert commands
    assert commands[0][-1] == "Transcribing…"
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
