"""Validated SpeakTap configuration with environment-variable loading."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .profiles import DEFAULT_PROFILE_ID, AsrModelProfile, get_asr_profile
from .runtime import installed_config_path

_DEFAULT_PROFILE = get_asr_profile(DEFAULT_PROFILE_ID)


def _parse_bool(value: str, *, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")


@dataclass(frozen=True, slots=True)
class SpeakTapConfig:
    speech_detector: str = "adaptive_energy"
    asr_profile: str = DEFAULT_PROFILE_ID
    asr_adapter: str = _DEFAULT_PROFILE.adapter
    asr_model: str = _DEFAULT_PROFILE.model
    asr_quantization: str = _DEFAULT_PROFILE.quantization
    execution_provider: str = "CPUExecutionProvider"
    asr_threads: int = 0
    asr_inter_op_threads: int = 1
    asr_execution_mode: str = "parallel"
    language: str = ""
    audio_device: str = "default"
    sample_rate: int = 16_000
    frame_milliseconds: int = 20
    speech_padding_milliseconds: int = 500
    max_recording_seconds: float = 300.0
    min_chunk_seconds: float = 4.0
    target_chunk_seconds: float = 15.0
    max_chunk_seconds: float = 25.0
    silence_milliseconds: int = 600
    forced_cut_overlap_milliseconds: int = 400
    max_pending_chunks: int = 8
    cleanup_enabled: bool = True
    cleanup_adapter: str = "safe"
    cleanup_model: str = ""
    cleanup_timeout_seconds: float = 2.0
    outputs: tuple[str, ...] = ("typing", "clipboard", "notification")
    sounds_enabled: bool = True
    # Capture is already live when the start cue plays, so the microphone can
    # record the cue itself. That is the deliberate trade for opening the
    # device first: the cue length is now overlap with the first words rather
    # than audio lost before them. The stop cue is unaffected, it only plays
    # once capture has stopped.
    start_sound: str = "/usr/share/sounds/freedesktop/stereo/service-login.oga"
    stop_sound: str = "/usr/share/sounds/freedesktop/stereo/service-logout.oga"

    def __post_init__(self) -> None:
        if not self.speech_detector:
            raise ValueError("speech_detector must not be empty")
        if not self.asr_profile or not self.asr_adapter:
            raise ValueError("asr_profile and asr_adapter must not be empty")
        profile = get_asr_profile(self.asr_profile)
        if (
            self.asr_adapter,
            self.asr_model,
            self.asr_quantization,
        ) != (
            profile.adapter,
            profile.model,
            profile.quantization,
        ):
            raise ValueError("ASR runtime fields must match the selected model profile")
        if self.sample_rate != profile.required_sample_rate:
            raise ValueError(
                f"profile {profile.profile_id!r} requires {profile.required_sample_rate} Hz audio"
            )
        if self.max_chunk_seconds > profile.max_chunk_seconds:
            raise ValueError(
                f"profile {profile.profile_id!r} supports chunks up to "
                f"{profile.max_chunk_seconds:g} seconds"
            )
        if not self.execution_provider:
            raise ValueError("execution_provider must not be empty")
        if self.asr_threads < 0 or self.asr_inter_op_threads < 0:
            raise ValueError("ASR thread counts must be non-negative")
        if self.asr_execution_mode not in {"sequential", "parallel"}:
            raise ValueError("asr_execution_mode must be sequential or parallel")
        if not self.audio_device:
            raise ValueError("audio_device must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_milliseconds <= 0:
            raise ValueError("frame_milliseconds must be positive")
        if self.speech_padding_milliseconds < 0:
            raise ValueError("speech_padding_milliseconds must be non-negative")
        if self.max_recording_seconds <= 0:
            raise ValueError("max_recording_seconds must be positive")
        if not 0 < self.min_chunk_seconds <= self.target_chunk_seconds <= self.max_chunk_seconds:
            raise ValueError("chunk durations must satisfy 0 < min <= target <= max")
        if self.silence_milliseconds <= 0:
            raise ValueError("silence_milliseconds must be positive")
        if not 0 <= self.forced_cut_overlap_milliseconds < self.max_chunk_seconds * 1000:
            raise ValueError("forced-cut overlap must fit inside the maximum chunk")
        if self.max_pending_chunks <= 0:
            raise ValueError("max_pending_chunks must be positive")
        if not self.cleanup_adapter:
            raise ValueError("cleanup_adapter must not be empty")
        if self.cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be positive")
        if not self.outputs or any(not output for output in self.outputs):
            raise ValueError("outputs must contain at least one non-empty adapter name")
        if self.sounds_enabled and (not self.start_sound or not self.stop_sound):
            raise ValueError("enabled sounds require start_sound and stop_sound")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        installed_profile: str | None = None,
    ) -> SpeakTapConfig:
        values = os.environ if env is None else env

        def get(name: str, default: str) -> str:
            return values.get(name, default).strip()

        default = cls()
        profile = get_asr_profile(
            get(
                "SPEAKTAP_PROFILE",
                installed_profile or default.asr_profile,
            )
        )
        outputs = tuple(
            output.strip()
            for output in get("SPEAKTAP_OUTPUTS", ",".join(default.outputs)).split(",")
            if output.strip()
        )
        return cls(
            speech_detector=get("SPEAKTAP_SPEECH_DETECTOR", default.speech_detector),
            asr_profile=profile.profile_id,
            asr_adapter=get("SPEAKTAP_ASR_ADAPTER", profile.adapter),
            asr_model=get("SPEAKTAP_ASR_MODEL", profile.model),
            asr_quantization=get("SPEAKTAP_ASR_QUANTIZATION", profile.quantization),
            execution_provider=get("SPEAKTAP_ONNX_EXECUTION_PROVIDER", default.execution_provider),
            asr_threads=int(get("SPEAKTAP_THREADS", str(default.asr_threads))),
            asr_inter_op_threads=int(
                get("SPEAKTAP_INTER_OP_THREADS", str(default.asr_inter_op_threads))
            ),
            asr_execution_mode=get("SPEAKTAP_EXECUTION_MODE", default.asr_execution_mode),
            language=get("SPEAKTAP_LANGUAGE", default.language),
            audio_device=get("SPEAKTAP_AUDIO_DEVICE", default.audio_device),
            sample_rate=int(get("SPEAKTAP_SAMPLE_RATE", str(default.sample_rate))),
            frame_milliseconds=int(
                get("SPEAKTAP_FRAME_MILLISECONDS", str(default.frame_milliseconds))
            ),
            speech_padding_milliseconds=int(
                get(
                    "SPEAKTAP_SPEECH_PADDING_MILLISECONDS",
                    str(default.speech_padding_milliseconds),
                )
            ),
            max_recording_seconds=float(
                get("SPEAKTAP_MAX_RECORDING_SECONDS", str(default.max_recording_seconds))
            ),
            min_chunk_seconds=float(
                get("SPEAKTAP_MIN_CHUNK_SECONDS", str(default.min_chunk_seconds))
            ),
            target_chunk_seconds=float(
                get("SPEAKTAP_TARGET_CHUNK_SECONDS", str(default.target_chunk_seconds))
            ),
            max_chunk_seconds=float(
                get("SPEAKTAP_MAX_CHUNK_SECONDS", str(default.max_chunk_seconds))
            ),
            silence_milliseconds=int(
                get("SPEAKTAP_SILENCE_MILLISECONDS", str(default.silence_milliseconds))
            ),
            forced_cut_overlap_milliseconds=int(
                get(
                    "SPEAKTAP_FORCED_CUT_OVERLAP_MILLISECONDS",
                    str(default.forced_cut_overlap_milliseconds),
                )
            ),
            max_pending_chunks=int(
                get("SPEAKTAP_MAX_PENDING_CHUNKS", str(default.max_pending_chunks))
            ),
            cleanup_enabled=_parse_bool(
                get("SPEAKTAP_CLEANUP_ENABLED", str(default.cleanup_enabled)),
                key="SPEAKTAP_CLEANUP_ENABLED",
            ),
            cleanup_adapter=get("SPEAKTAP_CLEANUP_ADAPTER", default.cleanup_adapter),
            cleanup_model=get("SPEAKTAP_CLEANUP_MODEL", default.cleanup_model),
            cleanup_timeout_seconds=float(
                get(
                    "SPEAKTAP_CLEANUP_TIMEOUT_SECONDS",
                    str(default.cleanup_timeout_seconds),
                )
            ),
            outputs=outputs,
            sounds_enabled=_parse_bool(
                get("SPEAKTAP_SOUNDS_ENABLED", str(default.sounds_enabled)),
                key="SPEAKTAP_SOUNDS_ENABLED",
            ),
            start_sound=get("SPEAKTAP_START_SOUND", default.start_sound),
            stop_sound=get("SPEAKTAP_STOP_SOUND", default.stop_sound),
        )

    @classmethod
    def load(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        path: Path | None = None,
    ) -> SpeakTapConfig:
        config_path = installed_config_path() if path is None else path
        installed_profile: str | None = None
        if config_path.exists():
            try:
                body: Any = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot read installed config: {error}") from error
            if not isinstance(body, dict):
                raise ValueError("installed config must be a JSON object")
            raw_profile = body.get("asr_profile")
            if not isinstance(raw_profile, str) or not raw_profile:
                raise ValueError("installed config requires a non-empty asr_profile")
            installed_profile = raw_profile
        runtime_values = dict(os.environ if env is None else env)
        runtime_values.pop("SPEAKTAP_PROFILE", None)
        return cls.from_env(
            runtime_values,
            installed_profile=installed_profile or DEFAULT_PROFILE_ID,
        )

    @property
    def model_profile(self) -> AsrModelProfile:
        return get_asr_profile(self.asr_profile)

    def with_profile(self, profile_id: str) -> SpeakTapConfig:
        profile = get_asr_profile(profile_id)
        return replace(
            self,
            asr_profile=profile.profile_id,
            asr_adapter=profile.adapter,
            asr_model=profile.model,
            asr_quantization=profile.quantization,
        )
