from __future__ import annotations

from pathlib import Path
from typing import Any

from speaktap.config import SpeakTapConfig
from speaktap.service_app import AsrService


def _service(monkeypatch: Any, tmp_path: Path) -> AsrService:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    return AsrService(SpeakTapConfig(outputs=("typing", "clipboard")))


def test_completion_message_announces_failed_segments(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    service = _service(monkeypatch, tmp_path)

    message = service._completion_message(
        "chunk 1",
        (),
        350,
        chunk_errors=("chunk 0: decoder unavailable",),
    )

    assert "failed segment" in message
    assert "typed" not in message


def test_completion_message_reports_success_when_no_segment_failed(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    service = _service(monkeypatch, tmp_path)

    message = service._completion_message("chunk 1", (), 350)

    assert message == "Transcribed and typed (350ms)."
