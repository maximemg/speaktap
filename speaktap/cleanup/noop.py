"""Raw-text fallback cleaner."""

from __future__ import annotations

from ..domain import CleanupResult, CleanupStatus, TranscriptChunk


class NoopCleaner:
    cleaner_id = "noop"

    def clean(
        self,
        assembled_text: str,
        chunks: tuple[TranscriptChunk, ...],
        *,
        timeout_seconds: float,
    ) -> CleanupResult:
        del chunks, timeout_seconds
        return CleanupResult(
            text=assembled_text,
            status=CleanupStatus.DISABLED,
            cleaner_id=self.cleaner_id,
        )

    def close(self) -> None:
        return None
