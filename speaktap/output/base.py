"""Desktop-independent output contract."""

from __future__ import annotations

from typing import Protocol

from ..domain import TranscriptionResult


class OutputAdapter(Protocol):
    output_id: str

    def deliver(self, result: TranscriptionResult) -> None: ...

    def close(self) -> None: ...
