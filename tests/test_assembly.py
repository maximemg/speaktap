from __future__ import annotations

from speaktap.assembly import assemble_transcript
from speaktap.domain import CutReason, TranscriptChunk


def _chunk(sequence: int, text: str) -> TranscriptChunk:
    return TranscriptChunk(
        session_id="session",
        sequence=sequence,
        audio_start_ms=sequence * 1_000,
        audio_end_ms=(sequence + 1) * 1_000,
        cut_reason=CutReason.FORCED_MAX,
        raw_text=text,
        asr_duration_ms=10,
    )


def test_orders_chunks_and_removes_overlap_case_insensitively() -> None:
    text = assemble_transcript(
        (
            _chunk(1, "brown fox jumps."),
            _chunk(0, "The quick Brown fox"),
        )
    )

    assert text == "The quick Brown fox jumps."


def test_empty_chunks_do_not_add_spacing() -> None:
    assert assemble_transcript((_chunk(0, ""), _chunk(1, "hello"))) == "hello"
