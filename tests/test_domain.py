from __future__ import annotations

import pytest

from speaktap.domain import (
    AudioFrame,
    CleanupStatus,
    CutReason,
    TranscriptChunk,
    TranscriptionResult,
)


def _chunk(sequence: int, *, text: str, error: str | None = None) -> TranscriptChunk:
    return TranscriptChunk(
        session_id="session",
        sequence=sequence,
        audio_start_ms=sequence * 1_000,
        audio_end_ms=(sequence + 1) * 1_000,
        cut_reason=CutReason.SILENCE,
        raw_text=text,
        asr_duration_ms=1,
        error=error,
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


def test_chunk_errors_report_failed_segments_in_sequence_order() -> None:
    result = TranscriptionResult(
        session_id="session",
        raw_text="first third",
        cleaned_text=None,
        cleanup_status=CleanupStatus.DISABLED,
        chunks=(
            _chunk(2, text="third"),
            _chunk(1, text="", error="backend exploded"),
            _chunk(0, text="first"),
        ),
    )

    assert result.chunk_errors == ("chunk 1: backend exploded",)


def test_chunk_errors_are_empty_when_every_segment_transcribed() -> None:
    result = TranscriptionResult(
        session_id="session",
        raw_text="first second",
        cleaned_text=None,
        cleanup_status=CleanupStatus.DISABLED,
        chunks=(_chunk(0, text="first"), _chunk(1, text="second")),
    )

    assert result.chunk_errors == ()


def test_successful_cleanup_requires_text() -> None:
    with pytest.raises(ValueError, match="requires cleaned_text"):
        TranscriptionResult(
            session_id="session",
            raw_text="raw",
            cleaned_text=None,
            cleanup_status=CleanupStatus.SUCCESS,
            chunks=(),
        )
