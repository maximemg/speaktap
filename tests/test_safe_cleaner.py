from __future__ import annotations

import pytest

from speaktap.cleanup.noop import NoopCleaner
from speaktap.cleanup.safe import SafeCleaner, safe_cleanup_text
from speaktap.config import SpeakTapConfig
from speaktap.domain import (
    AsrCapabilities,
    AsrResult,
    AudioChunk,
    CleanupStatus,
)
from speaktap.factory import make_pipeline


class UnusedBackend:
    backend_id = "unused"

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(16_000, 1, 25)

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def transcribe(self, chunk: AudioChunk, *, language: str = "") -> AsrResult:
        del chunk, language
        return AsrResult(text="", backend_id=self.backend_id)

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Um, I I think we should uh ship this today.",
            "I think we should ship this today.",
        ),
        (
            "euh je vais je vais partir demain",
            "je vais partir demain",
        ),
        (
            "we need to we need to keep HTTP 429",
            "we need to keep HTTP 429",
        ),
        (
            "send it Thursday no sorry send it Friday",
            "send it Thursday no sorry send it Friday",
        ),
        (
            "use process_audio_chunk from api.example.com/v2",
            "use process_audio_chunk from api.example.com/v2",
        ),
        (
            "send ER notes and charge the 20 Ah battery",
            "send ER notes and charge the 20 Ah battery",
        ),
        (
            "the umbrella is already clean",
            "the umbrella is already clean",
        ),
    ],
)
def test_safe_cleanup_text_applies_only_high_confidence_edits(
    raw: str,
    expected: str,
) -> None:
    assert safe_cleanup_text(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # French subject pronoun followed by the identical reflexive pronoun.
        "nous nous levons de bonne heure",
        "vous vous trompez souvent",
        # English doubling that grammar requires.
        "I had had enough of it",
        "I think that that argument fails",
        "what it is is a scheduling problem",
        # Conventional intensifiers and interjections.
        "it was very very cold",
        "we waited a long long time",
        "no no send it tomorrow",
    ],
)
def test_safe_cleanup_text_preserves_grammatical_doubling(raw: str) -> None:
    assert safe_cleanup_text(raw) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Doubling that is never grammatical stays removable.
        ("the the cat sat down", "the cat sat down"),
        ("I I think so", "I think so"),
        ("je vais je vais partir demain", "je vais partir demain"),
        # A longer stutter collapses onto the grammatical pair, not past it.
        ("nous nous nous nous levons", "nous nous levons"),
    ],
)
def test_safe_cleanup_text_still_removes_disfluent_repetition(
    raw: str,
    expected: str,
) -> None:
    assert safe_cleanup_text(raw) == expected


def test_safe_cleaner_returns_a_successful_cleanup_result() -> None:
    result = SafeCleaner().clean(
        "uh this this works",
        (),
        timeout_seconds=0.001,
    )

    assert result.text == "this works"
    assert result.status is CleanupStatus.SUCCESS
    assert result.cleaner_id == "safe"
    assert result.duration_ms >= 0


def test_factory_selects_safe_default_and_preserves_noop_escape_hatch() -> None:
    safe_pipeline = make_pipeline(SpeakTapConfig(), UnusedBackend())
    noop_pipeline = make_pipeline(
        SpeakTapConfig(cleanup_enabled=False, cleanup_adapter="noop"),
        UnusedBackend(),
    )

    assert isinstance(safe_pipeline._cleaner, SafeCleaner)
    assert isinstance(noop_pipeline._cleaner, NoopCleaner)
