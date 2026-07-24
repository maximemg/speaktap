"""Model-architecture-independent ASR contract."""

from __future__ import annotations

from typing import Protocol

from ..domain import AsrCapabilities, AsrResult, AudioChunk


class AsrBackend(Protocol):
    backend_id: str

    def capabilities(self) -> AsrCapabilities: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def transcribe(self, chunk: AudioChunk, *, language: str = "") -> AsrResult: ...

    def close(self) -> None: ...
