from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any

import pytest

from speaktap.capture import linux_arecord
from speaktap.capture.linux_arecord import ArecordSource

_FRAME_BYTES = 16_000 * 20 // 1000 * 2


class _FakeProcess:
    """Stand in for arecord with a real pipe so select() behaves normally."""

    def __init__(self, read_fd: int, stderr: bytes = b"") -> None:
        self.stdout = os.fdopen(read_fd, "rb")
        self.stderr = _FakeStderr(stderr)
        self.returncode: int | None = 0
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True


class _FakeStderr:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


def _install(
    monkeypatch: Any, stderr: bytes = b""
) -> tuple[ArecordSource, int, list[_FakeProcess]]:
    read_fd, write_fd = os.pipe()
    spawned: list[_FakeProcess] = []

    def spawn(*_args: object, **_kwargs: object) -> _FakeProcess:
        process = _FakeProcess(read_fd, stderr)
        spawned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    return ArecordSource(), write_fd, spawned


def test_priming_preserves_every_captured_byte(monkeypatch: Any) -> None:
    """Priming reads ahead of the buffered reader; nothing may be lost.

    start() pulls the first available bytes off the raw descriptor so it can
    return only once the device is live. Those bytes are held in the source,
    not in the stream, so a regression that forgets to replay them would drop
    or reorder the very beginning of the recording.
    """

    source, write_fd, _ = _install(monkeypatch)
    payload = bytes(index % 251 for index in range(3 * _FRAME_BYTES))
    os.write(write_fd, payload)
    source.start()
    os.close(write_fd)

    captured = b"".join(frame.pcm_s16le for frame in source.frames())

    assert captured == payload


def test_start_waits_for_the_device_to_deliver_audio(monkeypatch: Any) -> None:
    """The cue is played after start() returns, so start() must gate on audio."""

    source, write_fd, _ = _install(monkeypatch)
    delay_seconds = 0.2

    def feed() -> None:
        time.sleep(delay_seconds)
        os.write(write_fd, bytes(_FRAME_BYTES))

    writer = threading.Thread(target=feed)
    writer.start()
    started = time.monotonic()
    source.start()
    elapsed = time.monotonic() - started
    writer.join()
    os.close(write_fd)

    assert elapsed >= delay_seconds


def test_start_gives_up_on_a_device_that_never_delivers(monkeypatch: Any) -> None:
    """A wedged device must not hold the hotkey open forever."""

    monkeypatch.setattr(linux_arecord, "_PRIME_TIMEOUT_SECONDS", 0.1)
    source, write_fd, _ = _install(monkeypatch)

    started = time.monotonic()
    source.start()
    elapsed = time.monotonic() - started
    os.close(write_fd)

    assert elapsed < 1.0


def test_failed_open_surfaces_through_frames(monkeypatch: Any) -> None:
    """arecord closing stdout is an error, not a silent empty recording."""

    source, write_fd, spawned = _install(monkeypatch, stderr=b"cannot open device")
    os.close(write_fd)
    source.start()
    spawned[0].returncode = 1

    with pytest.raises(RuntimeError, match="cannot open device"):
        list(source.frames())
