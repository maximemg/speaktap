from __future__ import annotations

from speaktap.activity.adaptive_energy import AdaptiveEnergyDetector
from speaktap.domain import AudioFrame


def _frame(amplitude: int, start_ms: int = 0) -> AudioFrame:
    sample = amplitude.to_bytes(2, "little", signed=True)
    return AudioFrame(
        pcm_s16le=sample * 320,
        sample_rate=16_000,
        channels=1,
        start_ms=start_ms,
    )


def test_detects_speech_and_returns_to_silence() -> None:
    detector = AdaptiveEnergyDetector()

    assert detector.analyze(_frame(0)).is_speech is False
    assert detector.analyze(_frame(6_000, 20)).is_speech is True
    assert detector.analyze(_frame(0, 40)).is_speech is False


def test_reset_clears_hysteresis() -> None:
    detector = AdaptiveEnergyDetector()
    detector.analyze(_frame(6_000))
    detector.reset()

    activity = detector.analyze(_frame(0))

    assert activity.is_speech is False
    assert activity.noise_floor is not None
