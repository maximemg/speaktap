"""Finite PCM/WAV source used by safe end-to-end tests."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

from ..domain import AudioFrame


class AudioFileSource:
    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int = 16_000,
        frame_milliseconds: int = 20,
    ) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._frame_milliseconds = frame_milliseconds
        self._payload = b""
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("file source is already running")
        if self._path.suffix.casefold() == ".wav":
            self._payload = self._read_wav()
        else:
            self._payload = self._path.read_bytes()
        self._started = True

    def frames(self) -> Iterator[AudioFrame]:
        if not self._started:
            raise RuntimeError("file source has not been started")
        samples_per_frame = self._sample_rate * self._frame_milliseconds // 1000
        frame_bytes = samples_per_frame * 2
        for sequence, offset in enumerate(
            range(0, len(self._payload) - frame_bytes + 1, frame_bytes)
        ):
            yield AudioFrame(
                pcm_s16le=self._payload[offset : offset + frame_bytes],
                sample_rate=self._sample_rate,
                channels=1,
                start_ms=sequence * self._frame_milliseconds,
            )

    def stop(self) -> None:
        self._started = False
        self._payload = b""

    def _read_wav(self) -> bytes:
        with wave.open(str(self._path), "rb") as source:
            if source.getsampwidth() != 2:
                raise ValueError("WAV input must use 16-bit PCM")
            if source.getnchannels() != 1:
                raise ValueError("WAV input must be mono")
            if source.getframerate() != self._sample_rate:
                raise ValueError(
                    f"WAV input must be {self._sample_rate} Hz; got {source.getframerate()} Hz"
                )
            return source.readframes(source.getnframes())
