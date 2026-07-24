"""Core contracts for the SpeakTap streaming pipeline."""

from .config import SpeakTapConfig
from .domain import (
    AsrCapabilities,
    AsrResult,
    AudioChunk,
    AudioFrame,
    CleanupResult,
    CleanupStatus,
    CutReason,
    ServiceState,
    SessionOptions,
    TranscriptChunk,
    TranscriptionResult,
    WordTimestamp,
)
from .service import InvalidTransitionError, ServiceSnapshot, ServiceStateMachine

__all__ = [
    "AsrCapabilities",
    "AsrResult",
    "AudioChunk",
    "AudioFrame",
    "CleanupResult",
    "CleanupStatus",
    "CutReason",
    "InvalidTransitionError",
    "ServiceSnapshot",
    "ServiceState",
    "ServiceStateMachine",
    "SessionOptions",
    "SpeakTapConfig",
    "TranscriptChunk",
    "TranscriptionResult",
    "WordTimestamp",
]
