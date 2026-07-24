from __future__ import annotations

import json
from pathlib import Path

from speaktap.diagnostics import SessionLogger
from speaktap.domain import (
    CleanupStatus,
    CutReason,
    TranscriptChunk,
    TranscriptionResult,
)


def test_success_record_contains_chunk_timing_and_memory(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    result = TranscriptionResult(
        session_id="session",
        raw_text="hello",
        cleaned_text=None,
        cleanup_status=CleanupStatus.DISABLED,
        chunks=(
            TranscriptChunk(
                session_id="session",
                sequence=0,
                audio_start_ms=100,
                audio_end_ms=1_100,
                cut_reason=CutReason.STOP,
                raw_text="hello",
                asr_duration_ms=80,
                speech_ms=700,
                queue_wait_ms=4,
            ),
        ),
        timings_ms={"post_stop": 85},
    )

    SessionLogger(path).success(
        result,
        service_post_stop_ms=92,
        output_errors=(),
    )

    record = json.loads(path.read_text())
    assert record["status"] == "ok"
    assert record["audio_ms"] == 1_100
    assert record["speech_ms"] == 700
    assert record["timings_ms"]["service_post_stop"] == 92
    assert record["chunks"][0]["queue_wait_ms"] == 4
    assert "vmrss_kib" in record["memory"]


def test_failure_record_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "sessions.jsonl"
    logger = SessionLogger(path)

    logger.failure(
        session_id="one",
        error="capture failed",
        service_post_stop_ms=12,
    )
    logger.failure(
        session_id="two",
        error="model failed",
        service_post_stop_ms=34,
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["session_id"] for record in records] == ["one", "two"]
