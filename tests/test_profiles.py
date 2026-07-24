from __future__ import annotations

from pathlib import Path

import pytest

from speaktap.profiles import (
    DEFAULT_PROFILE_ID,
    LanguageMode,
    ProfileStatus,
    benchmark_profile_ids,
    get_asr_profile,
    list_asr_profiles,
)
from speaktap.transcription import OnnxAsrBackend


def test_registry_has_one_admitted_default_and_sequential_candidates() -> None:
    profiles = list_asr_profiles()

    assert profiles[0].profile_id == DEFAULT_PROFILE_ID
    assert profiles[0].status is ProfileStatus.SUPPORTED
    assert all(profile.adapter == "onnx_asr" for profile in profiles)
    assert all(len(profile.revision) == 40 for profile in profiles)
    assert all(profile.repository and profile.files for profile in profiles)
    assert all(
        len(profile.files) == len(profile.sha256)
        and all(len(digest) == 64 for digest in profile.sha256)
        for profile in profiles
    )
    assert benchmark_profile_ids() == tuple(profile.profile_id for profile in profiles)


def test_legacy_fast_alias_resolves_to_default() -> None:
    assert get_asr_profile("fast").profile_id == DEFAULT_PROFILE_ID


def test_candidate_capabilities_are_profile_specific() -> None:
    profile = get_asr_profile("canary-1b-v2-int8")
    backend = OnnxAsrBackend(profile=profile)

    capabilities = backend.capabilities()

    assert profile.language_mode is LanguageMode.REQUIRED_HINT
    assert capabilities.supports_language_hint is True
    assert capabilities.max_chunk_seconds == profile.max_chunk_seconds


def test_backend_rejects_a_model_artifact_with_the_wrong_checksum(tmp_path: Path) -> None:
    profile = get_asr_profile(DEFAULT_PROFILE_ID)
    for filename in profile.files:
        artifact = tmp_path / filename
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        OnnxAsrBackend(profile=profile)._verify_model_files(tmp_path)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ASR profile"):
        get_asr_profile("made-up-model")
