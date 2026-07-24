from __future__ import annotations

from pathlib import Path

import pytest

from speaktap.config import SpeakTapConfig
from speaktap.profiles import DEFAULT_PROFILE_ID


def test_defaults_are_valid() -> None:
    config = SpeakTapConfig()

    assert config.asr_profile == DEFAULT_PROFILE_ID
    assert config.execution_provider == "CPUExecutionProvider"
    assert config.cleanup_enabled is True
    assert config.cleanup_adapter == "safe"
    assert config.min_chunk_seconds < config.target_chunk_seconds < config.max_chunk_seconds


def test_environment_overrides() -> None:
    config = SpeakTapConfig.from_env(
        {
            "SPEAKTAP_PROFILE": "canary-1b-v2-int8",
            "SPEAKTAP_CLEANUP_ENABLED": "true",
            "SPEAKTAP_MAX_CHUNK_SECONDS": "30",
            "SPEAKTAP_OUTPUTS": "clipboard,notification",
            "SPEAKTAP_SOUNDS_ENABLED": "false",
        }
    )

    assert config.asr_profile == "canary-1b-v2-int8"
    assert config.asr_model == "nemo-canary-1b-v2"
    assert config.cleanup_enabled is True
    assert config.max_chunk_seconds == 30
    assert config.outputs == ("clipboard", "notification")
    assert config.sounds_enabled is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "SPEAKTAP_MIN_CHUNK_SECONDS": "20",
                "SPEAKTAP_TARGET_CHUNK_SECONDS": "10",
            },
            "chunk durations",
        ),
        ({"SPEAKTAP_CLEANUP_ENABLED": "sometimes"}, "must be a boolean"),
        ({"SPEAKTAP_OUTPUTS": ""}, "outputs"),
    ],
)
def test_invalid_environment_rejected(overrides: dict[str, str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SpeakTapConfig.from_env(overrides)


def test_installed_profile_cannot_be_overridden_at_runtime(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"asr_profile":"canary-1b-v2-int8"}\n')

    installed = SpeakTapConfig.load({}, path=path)
    attempted_override = SpeakTapConfig.load(
        {"SPEAKTAP_PROFILE": "whisper-large-v3-turbo-int8"},
        path=path,
    )

    assert installed.asr_model == "nemo-canary-1b-v2"
    assert attempted_override.asr_model == "nemo-canary-1b-v2"


def test_profile_runtime_fields_cannot_drift() -> None:
    with pytest.raises(ValueError, match="must match"):
        SpeakTapConfig.from_env(
            {
                "SPEAKTAP_PROFILE": "canary-1b-v2-int8",
                "SPEAKTAP_ASR_MODEL": "nemo-parakeet-tdt-0.6b-v3",
            }
        )
