from __future__ import annotations

from speaktap.cleanup.noop import NoopCleaner
from speaktap.domain import CleanupStatus


def test_noop_cleaner_returns_raw_text() -> None:
    cleaner = NoopCleaner()

    result = cleaner.clean("keep this", (), timeout_seconds=1)

    assert result.text == "keep this"
    assert result.status is CleanupStatus.DISABLED
    assert result.cleaner_id == "noop"
