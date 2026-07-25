"""Runtime-independent domain objects shared by SpeakTap pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

type DiagnosticValue = str | int | float | bool | None


class ServiceState(StrEnum):
    """Externally observable service states."""

    IDLE = "idle"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    OUTPUTTING = "outputting"


class CutReason(StrEnum):
    """Why an audio chunk was finalized."""

    SILENCE = "silence"
    FORCED_MAX = "forced_max"
    STOP = "stop"


class CleanupStatus(StrEnum):
    """Outcome of the optional transcript cleanup stage."""

    DISABLED = "disabled"
    SUCCESS = "success"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One timestamped PCM S16_LE frame from an audio source."""

    pcm_s16le: bytes
    sample_rate: int
    channels: int
    start_ms: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        frame_width = 2 * self.channels
        if not self.pcm_s16le:
            raise ValueError("pcm_s16le must not be empty")
        if len(self.pcm_s16le) % frame_width:
            raise ValueError("PCM byte length must align to complete S16_LE samples")

    @property
    def sample_count(self) -> int:
        """Return samples per channel."""

        return len(self.pcm_s16le) // (2 * self.channels)

    @property
    def duration_ms(self) -> float:
        return self.sample_count * 1000 / self.sample_rate

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A finalized, ordered unit of PCM audio for ASR."""

    session_id: str
    sequence: int
    pcm_s16le: bytes
    sample_rate: int
    channels: int
    start_ms: int
    end_ms: int
    speech_ms: int
    cut_reason: CutReason
    overlap_ms: int = 0

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not self.pcm_s16le:
            raise ValueError("pcm_s16le must not be empty")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("chunk timestamps must describe a positive duration")
        duration_ms = self.end_ms - self.start_ms
        if not 0 <= self.speech_ms <= duration_ms:
            raise ValueError("speech_ms must fit inside the chunk duration")
        if not 0 <= self.overlap_ms < duration_ms:
            raise ValueError("overlap_ms must be smaller than the chunk duration")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("word text must not be empty")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("word timestamps are invalid")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AsrCapabilities:
    """Constraints and optional features advertised by an ASR adapter."""

    required_sample_rate: int
    preferred_chunk_seconds: float
    max_chunk_seconds: float
    supports_word_timestamps: bool = False
    supports_language_hint: bool = False
    supports_streaming: bool = False
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if self.required_sample_rate <= 0:
            raise ValueError("required_sample_rate must be positive")
        if self.preferred_chunk_seconds <= 0:
            raise ValueError("preferred_chunk_seconds must be positive")
        if self.max_chunk_seconds < self.preferred_chunk_seconds:
            raise ValueError("max_chunk_seconds must be at least the preferred duration")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")


@dataclass(frozen=True, slots=True)
class AsrResult:
    text: str
    backend_id: str
    words: tuple[WordTimestamp, ...] = ()
    diagnostics: Mapping[str, DiagnosticValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend_id:
            raise ValueError("backend_id must not be empty")


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    session_id: str
    sequence: int
    audio_start_ms: int
    audio_end_ms: int
    cut_reason: CutReason
    raw_text: str
    asr_duration_ms: int
    speech_ms: int = 0
    overlap_ms: int = 0
    queue_wait_ms: int = 0
    words: tuple[WordTimestamp, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.audio_start_ms < 0 or self.audio_end_ms <= self.audio_start_ms:
            raise ValueError("audio timestamps are invalid")
        if self.asr_duration_ms < 0:
            raise ValueError("asr_duration_ms must be non-negative")
        duration_ms = self.audio_end_ms - self.audio_start_ms
        if not 0 <= self.speech_ms <= duration_ms:
            raise ValueError("speech_ms must fit inside the audio duration")
        if not 0 <= self.overlap_ms < duration_ms:
            raise ValueError("overlap_ms must be smaller than the audio duration")
        if self.queue_wait_ms < 0:
            raise ValueError("queue_wait_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class CleanupResult:
    text: str
    status: CleanupStatus
    cleaner_id: str
    duration_ms: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.cleaner_id:
            raise ValueError("cleaner_id must not be empty")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if self.status is CleanupStatus.SUCCESS and self.error is not None:
            raise ValueError("successful cleanup cannot contain an error")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    session_id: str
    raw_text: str
    cleaned_text: str | None
    cleanup_status: CleanupStatus
    chunks: tuple[TranscriptChunk, ...]
    timings_ms: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.cleanup_status is CleanupStatus.SUCCESS and self.cleaned_text is None:
            raise ValueError("successful cleanup requires cleaned_text")
        if any(value < 0 for value in self.timings_ms.values()):
            raise ValueError("timings must be non-negative")

    @property
    def output_text(self) -> str:
        return self.cleaned_text if self.cleaned_text is not None else self.raw_text

    @property
    def chunk_errors(self) -> tuple[str, ...]:
        """Describe chunks whose transcription failed, in sequence order.

        A failed chunk contributes empty text, so assembly drops it and the
        transcript silently loses that span of speech. Callers must surface
        these before presenting the result as a complete transcript.
        """

        return tuple(
            f"chunk {chunk.sequence}: {chunk.error}"
            for chunk in sorted(self.chunks, key=lambda item: item.sequence)
            if chunk.error is not None
        )


@dataclass(frozen=True, slots=True)
class SessionOptions:
    asr_profile: str
    cleanup_enabled: bool

    def __post_init__(self) -> None:
        if not self.asr_profile:
            raise ValueError("asr_profile must not be empty")
