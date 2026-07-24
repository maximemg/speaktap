"""Speech activity detection contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..domain import AudioFrame


@dataclass(frozen=True, slots=True)
class SpeechActivity:
    is_speech: bool
    probability: float
    noise_floor: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        if self.noise_floor is not None and self.noise_floor < 0:
            raise ValueError("noise_floor must be non-negative")


class SpeechDetector(Protocol):
    detector_id: str

    def reset(self) -> None: ...

    def analyze(self, frame: AudioFrame) -> SpeechActivity: ...

    def close(self) -> None: ...
