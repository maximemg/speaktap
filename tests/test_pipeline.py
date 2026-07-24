from __future__ import annotations

from collections.abc import Iterator

from speaktap.activity.adaptive_energy import AdaptiveEnergyDetector
from speaktap.cleanup.noop import NoopCleaner
from speaktap.cleanup.safe import SafeCleaner
from speaktap.domain import (
    AsrCapabilities,
    AsrResult,
    AudioChunk,
    AudioFrame,
    CleanupStatus,
)
from speaktap.pipeline import PipelineSession
from speaktap.segmentation.adaptive import AdaptiveChunkPolicy


class FiniteSource:
    def __init__(self, frames: tuple[AudioFrame, ...]) -> None:
        self._frames = frames

    def start(self) -> None:
        return None

    def frames(self) -> Iterator[AudioFrame]:
        yield from self._frames

    def stop(self) -> None:
        return None


class FakeBackend:
    backend_id = "fake"

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(16_000, 1, 2)

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def transcribe(self, chunk: AudioChunk, *, language: str = "") -> AsrResult:
        del language
        return AsrResult(text=f"chunk {chunk.sequence}", backend_id=self.backend_id)

    def close(self) -> None:
        return None


class DisfluentBackend(FakeBackend):
    def transcribe(self, chunk: AudioChunk, *, language: str = "") -> AsrResult:
        del chunk, language
        return AsrResult(
            text="um I I think this works",
            backend_id=self.backend_id,
        )


def _speech_frames(count: int) -> tuple[AudioFrame, ...]:
    sample = (6_000).to_bytes(2, "little", signed=True)
    return tuple(
        AudioFrame(
            pcm_s16le=sample * 320,
            sample_rate=16_000,
            channels=1,
            start_ms=index * 20,
        )
        for index in range(count)
    )


def test_finite_source_runs_through_chunking_asr_and_assembly() -> None:
    pipeline = PipelineSession(
        detector=AdaptiveEnergyDetector(),
        chunk_policy=AdaptiveChunkPolicy(
            min_chunk_seconds=0.1,
            target_chunk_seconds=0.2,
            max_chunk_seconds=0.3,
            silence_milliseconds=40,
            padding_milliseconds=20,
            forced_cut_overlap_milliseconds=40,
        ),
        backend=FakeBackend(),
        cleaner=NoopCleaner(),
        max_recording_seconds=10,
    )
    pipeline.start(
        FiniteSource(_speech_frames(20)),
        session_id="session",
        cleanup_enabled=False,
    )

    result = pipeline.wait()

    assert result.output_text == "chunk 0 chunk 1"
    assert len(result.chunks) == 2
    assert result.cleanup_status is CleanupStatus.DISABLED


def test_enabled_safe_cleanup_runs_once_after_final_assembly() -> None:
    pipeline = PipelineSession(
        detector=AdaptiveEnergyDetector(),
        chunk_policy=AdaptiveChunkPolicy(
            min_chunk_seconds=0.1,
            target_chunk_seconds=0.2,
            max_chunk_seconds=0.3,
            silence_milliseconds=40,
            padding_milliseconds=20,
            forced_cut_overlap_milliseconds=40,
        ),
        backend=DisfluentBackend(),
        cleaner=SafeCleaner(),
        max_recording_seconds=10,
    )
    pipeline.start(
        FiniteSource(_speech_frames(6)),
        session_id="session",
        cleanup_enabled=True,
    )

    result = pipeline.wait()

    assert result.raw_text == "um I I think this works"
    assert result.cleaned_text == "I think this works"
    assert result.output_text == "I think this works"
    assert result.cleanup_status is CleanupStatus.SUCCESS
