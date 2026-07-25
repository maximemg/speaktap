from __future__ import annotations

import threading
import time
from typing import Any

from speaktap.profiles import AsrModelProfile, LanguageMode, ProfileStatus
from speaktap.transcription.onnx_backend import OnnxAsrBackend


def _profile(*, sample_rate: int) -> AsrModelProfile:
    return AsrModelProfile(
        profile_id="test-profile",
        display_name="Test profile",
        status=ProfileStatus.CANDIDATE,
        architecture="test",
        adapter="onnx_asr",
        model="test-model",
        repository="example/test-model",
        revision="a" * 40,
        files=("model.onnx",),
        sha256=("b" * 64,),
        quantization="int8",
        languages=("en",),
        language_mode=LanguageMode.AUTOMATIC,
        required_sample_rate=sample_rate,
        preferred_chunk_seconds=10.0,
        max_chunk_seconds=20.0,
        supports_word_timestamps=False,
        expected_memory_mib=None,
        notes="test fixture",
    )


class RecordingModel:
    def __init__(self) -> None:
        self.sample_rates: list[int] = []

    def recognize(self, waveform: Any, sample_rate: int, **_kwargs: Any) -> str:
        del waveform
        self.sample_rates.append(sample_rate)
        return ""


class CountingBackend(OnnxAsrBackend):
    """Replace the download-and-load step with a slow, counted stub."""

    def __init__(self, profile: AsrModelProfile) -> None:
        super().__init__(profile=profile)
        self.build_count = 0

    def _build_model(self) -> Any:
        # Widen the window so an unsynchronised check-then-act loses the race.
        time.sleep(0.05)
        self.build_count += 1
        return RecordingModel()


def test_warmup_uses_the_sample_rate_the_profile_requires() -> None:
    backend = OnnxAsrBackend(profile=_profile(sample_rate=8_000))
    model = RecordingModel()
    backend._model = model

    backend.warmup()

    assert model.sample_rates == [8_000]


def test_concurrent_load_builds_the_model_exactly_once() -> None:
    backend = CountingBackend(_profile(sample_rate=16_000))
    thread_count = 8
    start = threading.Barrier(thread_count)

    def worker() -> None:
        start.wait()
        backend.load()

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert backend.build_count == 1
