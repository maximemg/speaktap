"""Streaming audio segmentation contract."""

from __future__ import annotations

from typing import Protocol

from ..activity.base import SpeechActivity
from ..domain import AudioChunk, AudioFrame


class ChunkPolicy(Protocol):
    def reset(self, session_id: str) -> None: ...

    def push(
        self,
        frame: AudioFrame,
        activity: SpeechActivity,
    ) -> tuple[AudioChunk, ...]: ...

    def finish(self) -> tuple[AudioChunk, ...]: ...
