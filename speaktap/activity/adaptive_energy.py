"""Low-cost adaptive energy speech detector for the initial CPU pipeline."""

from __future__ import annotations

import math

import numpy as np

from ..domain import AudioFrame
from .base import SpeechActivity


class AdaptiveEnergyDetector:
    """Detect speech from RMS energy with an adaptive noise floor and hysteresis."""

    detector_id = "adaptive_energy"

    def __init__(
        self,
        *,
        initial_noise_floor: float = 0.0005,
        speech_ratio: float = 3.0,
        silence_ratio: float = 1.8,
        minimum_speech_rms: float = 0.0025,
        minimum_silence_rms: float = 0.0015,
    ) -> None:
        self._initial_noise_floor = initial_noise_floor
        self._speech_ratio = speech_ratio
        self._silence_ratio = silence_ratio
        self._minimum_speech_rms = minimum_speech_rms
        self._minimum_silence_rms = minimum_silence_rms
        self.reset()

    def reset(self) -> None:
        self._noise_floor = self._initial_noise_floor
        self._in_speech = False
        self._initialized = False

    def analyze(self, frame: AudioFrame) -> SpeechActivity:
        samples = np.frombuffer(frame.pcm_s16le, dtype="<i2").astype(np.float32)
        samples *= 1.0 / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        if not self._initialized:
            # The first frame is normally the brief pause after the start cue.
            # Cap its influence so immediate speech cannot become the noise floor.
            self._noise_floor = max(
                self._initial_noise_floor,
                min(rms, 0.005),
            )
            self._initialized = True

        start_threshold = max(
            self._minimum_speech_rms,
            self._noise_floor * self._speech_ratio,
        )
        stop_threshold = max(
            self._minimum_silence_rms,
            self._noise_floor * self._silence_ratio,
        )
        threshold = stop_threshold if self._in_speech else start_threshold
        self._in_speech = rms >= threshold

        if not self._in_speech:
            # Track a changing room/noise level without letting speech rapidly
            # raise the floor and suppress following words.
            alpha = 0.98 if rms > self._noise_floor else 0.90
            target = min(
                rms,
                max(self._initial_noise_floor, self._noise_floor * 1.5),
            )
            self._noise_floor = alpha * self._noise_floor + (1.0 - alpha) * target

        ratio = rms / max(threshold, 1e-9)
        probability = 1.0 - math.exp(-ratio)
        probability = max(0.5, probability) if self._in_speech else min(0.499, probability)
        return SpeechActivity(
            is_speech=self._in_speech,
            probability=max(0.0, min(1.0, probability)),
            noise_floor=self._noise_floor,
        )

    def close(self) -> None:
        return None
