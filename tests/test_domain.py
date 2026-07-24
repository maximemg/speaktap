from __future__ import annotations

import pytest

from speaktap.domain import (
    AudioFrame,
    CleanupStatus,
    TranscriptionResult,
)


def test_audio_frame_duration() -> None:
    frame = AudioFrame(
        pcm_s16le=b"\0\0" * 320,
        sample_rate=16_000,
        channels=1,
        start_ms=100,
    )

    assert frame.sample_count == 320
    assert frame.duration_ms == 20
    assert frame.end_ms == 120


def test_audio_frame_rejects_partial_sample() -> None:
    with pytest.raises(ValueError, match="align"):
        AudioFrame(pcm_s16le=b"\0", sample_rate=16_000, channels=1, start_ms=0)


def test_transcription_output_prefers_clean_text() -> None:
    result = TranscriptionResult(
        session_id="session",
        raw_text="raw",
        cleaned_text="clean",
        cleanup_status=CleanupStatus.SUCCESS,
        chunks=(),
    )

    assert result.output_text == "clean"


def test_successful_cleanup_requires_text() -> None:
    with pytest.raises(ValueError, match="requires cleaned_text"):
        TranscriptionResult(
            session_id="session",
            raw_text="raw",
            cleaned_text=None,
            cleanup_status=CleanupStatus.SUCCESS,
            chunks=(),
        )
