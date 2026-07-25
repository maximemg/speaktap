"""ALSA capture backed by the ubiquitous arecord command."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from typing import BinaryIO, cast

from ..domain import AudioFrame


class ArecordSource:
    def __init__(
        self,
        *,
        device: str = "default",
        sample_rate: int = 16_000,
        frame_milliseconds: int = 20,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._frame_milliseconds = frame_milliseconds
        self._process: subprocess.Popen[bytes] | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("arecord source is already running")
        self._stopping.clear()
        self._process = subprocess.Popen(
            [
                "arecord",
                "-q",
                "-D",
                self._device,
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(self._sample_rate),
                "-c",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def frames(self) -> Iterator[AudioFrame]:
        process = self._require_process()
        if process.stdout is None:
            raise RuntimeError("arecord stdout is unavailable")
        samples_per_frame = self._sample_rate * self._frame_milliseconds // 1000
        frame_bytes = samples_per_frame * 2
        sequence = 0
        while not self._stopping.is_set():
            payload = self._read_exact(cast(BinaryIO, process.stdout), frame_bytes)
            if len(payload) != frame_bytes:
                break
            # Raw arecord output carries no timestamps. Positions are nominal,
            # synthesized from complete frame count; ALSA underruns therefore
            # are not represented as wall-clock gaps in start_ms.
            yield AudioFrame(
                pcm_s16le=payload,
                sample_rate=self._sample_rate,
                channels=1,
                start_ms=sequence * self._frame_milliseconds,
            )
            sequence += 1
        return_code = process.poll()
        if (
            not self._stopping.is_set()
            and return_code not in (None, 0)
            and process.stderr is not None
        ):
            detail = process.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"arecord exited with status {return_code}: {detail}")

    def stop(self) -> None:
        self._stopping.set()
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        self._process = None

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("arecord source has not been started")
        return self._process

    @staticmethod
    def _read_exact(stream: BinaryIO, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
