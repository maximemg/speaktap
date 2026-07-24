"""Construct the SpeakTap pipeline from validated configuration."""

from __future__ import annotations

from .activity import AdaptiveEnergyDetector
from .cleanup import Cleaner, NoopCleaner, SafeCleaner
from .config import SpeakTapConfig
from .pipeline import PipelineSession
from .segmentation import AdaptiveChunkPolicy
from .transcription.base import AsrBackend


def make_pipeline(config: SpeakTapConfig, backend: AsrBackend) -> PipelineSession:
    if config.speech_detector != "adaptive_energy":
        raise ValueError(f"speech detector {config.speech_detector!r} is not implemented")
    cleaner: Cleaner
    if config.cleanup_adapter == "safe":
        cleaner = SafeCleaner()
    elif config.cleanup_adapter == "noop":
        cleaner = NoopCleaner()
    else:
        raise ValueError(f"cleanup adapter {config.cleanup_adapter!r} is not implemented")
    return PipelineSession(
        detector=AdaptiveEnergyDetector(),
        chunk_policy=AdaptiveChunkPolicy(
            min_chunk_seconds=config.min_chunk_seconds,
            target_chunk_seconds=config.target_chunk_seconds,
            max_chunk_seconds=config.max_chunk_seconds,
            silence_milliseconds=config.silence_milliseconds,
            padding_milliseconds=config.speech_padding_milliseconds,
            forced_cut_overlap_milliseconds=(config.forced_cut_overlap_milliseconds),
        ),
        backend=backend,
        cleaner=cleaner,
        language=config.language,
        cleanup_timeout_seconds=config.cleanup_timeout_seconds,
        max_pending_chunks=config.max_pending_chunks,
        max_recording_seconds=config.max_recording_seconds,
    )
