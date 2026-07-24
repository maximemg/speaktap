"""Transcript cleanup contract."""

from __future__ import annotations

from typing import Protocol

from ..domain import CleanupResult, TranscriptChunk


class Cleaner(Protocol):
    cleaner_id: str

    def clean(
        self,
        assembled_text: str,
        chunks: tuple[TranscriptChunk, ...],
        *,
        timeout_seconds: float,
    ) -> CleanupResult: ...

    def close(self) -> None: ...
