"""Audio capture contract."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from ..domain import AudioFrame


class AudioSource(Protocol):
    def start(self) -> None: ...

    def frames(self) -> Iterator[AudioFrame]: ...

    def stop(self) -> None: ...
