"""ALSA capture backed by the ubiquitous arecord command."""

from __future__ import annotations

import os
import select
import subprocess
import threading
from collections.abc import Iterator
from typing import BinaryIO, cast

from ..domain import AudioFrame

# Opening the ALSA device costs ~100-150 ms locally before the first sample
# arrives. start() absorbs that wait so callers can cue the speaker against a
# live microphone, but it must never hang the hotkey on a wedged device.
_PRIME_TIMEOUT_SECONDS = 1.0


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
        self._pending = b""

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("arecord source is already running")
        self._stopping.clear()
        self._pending = b""
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
        if self._process.stdout is not None:
            self._prime(cast(BinaryIO, self._process.stdout))

    def _prime(self, stream: BinaryIO) -> None:
        """Return once the device is delivering audio, or the timeout expires.

        A failed open closes stdout instead of writing to it, which select
        reports as readable and os.read reports as EOF, so the error still
        surfaces through frames() rather than stalling here for a second.
        Bytes are read from the raw descriptor while the buffered reader is
        still empty, so holding them here preserves stream order.
        """

        readable, _, _ = select.select([stream], [], [], _PRIME_TIMEOUT_SECONDS)
        if not readable:
            return
        samples_per_frame = self._sample_rate * self._frame_milliseconds // 1000
        self._pending = os.read(stream.fileno(), samples_per_frame * 2)

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
        self._pending = b""
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

    def _read_exact(self, stream: BinaryIO, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        if self._pending:
            chunks.append(self._pending[:remaining])
            self._pending = self._pending[remaining:]
            remaining -= len(chunks[0])
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
