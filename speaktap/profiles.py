"""Explicit ASR model profiles admitted to the CPU/ONNX runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ProfileStatus(StrEnum):
    SUPPORTED = "supported"
    CANDIDATE = "candidate"


class LanguageMode(StrEnum):
    AUTOMATIC = "automatic"
    OPTIONAL_HINT = "optional_hint"
    REQUIRED_HINT = "required_hint"


@dataclass(frozen=True, slots=True)
class AsrModelProfile:
    profile_id: str
    display_name: str
    status: ProfileStatus
    architecture: str
    adapter: str
    model: str
    repository: str
    revision: str
    files: tuple[str, ...]
    sha256: tuple[str, ...]
    quantization: str
    languages: tuple[str, ...]
    language_mode: LanguageMode
    required_sample_rate: int
    preferred_chunk_seconds: float
    max_chunk_seconds: float
    supports_word_timestamps: bool
    expected_memory_mib: int | None
    notes: str

    def __post_init__(self) -> None:
        if not self.profile_id or not self.display_name:
            raise ValueError("profile ID and display name must not be empty")
        if not self.adapter or not self.model or not self.quantization:
            raise ValueError("profile runtime fields must not be empty")
        if not self.repository or "/" not in self.repository:
            raise ValueError("profile model repository must be a Hugging Face repository ID")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("profile model revision must be a full lowercase commit SHA")
        if not self.files or any(not filename for filename in self.files):
            raise ValueError("profile model files must not be empty")
        if len(self.sha256) != len(self.files) or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.sha256
        ):
            raise ValueError("profile SHA-256 values must match every model file")
        if not self.languages or any(not language for language in self.languages):
            raise ValueError("profile languages must not be empty")
        if self.required_sample_rate <= 0:
            raise ValueError("profile sample rate must be positive")
        if not 0 < self.preferred_chunk_seconds <= self.max_chunk_seconds:
            raise ValueError("profile chunk durations are invalid")
        if self.expected_memory_mib is not None and self.expected_memory_mib <= 0:
            raise ValueError("expected profile memory must be positive")

    @property
    def supports_language_hint(self) -> bool:
        return self.language_mode is not LanguageMode.AUTOMATIC

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


_EUROPEAN_LANGUAGES = (
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
)

DEFAULT_PROFILE_ID = "parakeet-tdt-v3-int8"

_PROFILES = (
    AsrModelProfile(
        profile_id=DEFAULT_PROFILE_ID,
        display_name="Parakeet TDT 0.6B v3 int8",
        status=ProfileStatus.SUPPORTED,
        architecture="tdt_transducer",
        adapter="onnx_asr",
        model="nemo-parakeet-tdt-0.6b-v3",
        repository="istupakov/parakeet-tdt-0.6b-v3-onnx",
        revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        files=(
            "config.json",
            "decoder_joint-model.int8.onnx",
            "encoder-model.int8.onnx",
            "vocab.txt",
        ),
        sha256=(
            "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
            "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
            "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
            "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
        ),
        quantization="int8",
        languages=_EUROPEAN_LANGUAGES,
        language_mode=LanguageMode.AUTOMATIC,
        required_sample_rate=16_000,
        preferred_chunk_seconds=15.0,
        max_chunk_seconds=30.0,
        supports_word_timestamps=False,
        expected_memory_mib=1_250,
        notes="Current fast multilingual baseline; locally benchmarked and admitted.",
    ),
    AsrModelProfile(
        profile_id="canary-1b-v2-int8",
        display_name="Canary 1B v2 int8",
        status=ProfileStatus.CANDIDATE,
        architecture="attention_encoder_decoder",
        adapter="onnx_asr",
        model="nemo-canary-1b-v2",
        repository="istupakov/canary-1b-v2-onnx",
        revision="5ebc1520cef7b6b318b3526ad17adbfe00bc1bfc",
        files=(
            "config.json",
            "decoder-model.int8.onnx",
            "encoder-model.int8.onnx",
            "vocab.txt",
        ),
        sha256=(
            "f90ace8e35326dcd47c7330b230644fa0835083ed1e89e2f59aa08ba10d74f54",
            "52d83aa7aad41fbbe4f9dfcd341d784735a6eb4c6eb0d3290fc27a0d8ac39abf",
            "6d96e9945898e5ace48f4efecd459ca1df81859730be27b8af6b197639403ee1",
            "2c9efe6104fd29522ea27ce0e3aef5d37c690af4e5a4232e643e23ca403ffea3",
        ),
        quantization="int8",
        languages=_EUROPEAN_LANGUAGES,
        language_mode=LanguageMode.REQUIRED_HINT,
        required_sample_rate=16_000,
        preferred_chunk_seconds=15.0,
        max_chunk_seconds=30.0,
        supports_word_timestamps=False,
        expected_memory_mib=None,
        notes="Multilingual quality candidate; non-English use requires SPEAKTAP_LANGUAGE.",
    ),
    AsrModelProfile(
        profile_id="whisper-large-v3-turbo-int8",
        display_name="Whisper Large v3 Turbo int8",
        status=ProfileStatus.CANDIDATE,
        architecture="attention_encoder_decoder",
        adapter="onnx_asr",
        model="onnx-community/whisper-large-v3-turbo",
        repository="onnx-community/whisper-large-v3-turbo",
        revision="360ebcde2559d60bb474678be3c1de9ef347d01a",
        files=(
            "added_tokens.json",
            "config.json",
            "onnx/decoder_model_merged_int8.onnx",
            "onnx/encoder_model_int8.onnx",
            "vocab.json",
        ),
        sha256=(
            "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0",
            "35cd83669f75bc2867f3b3a4461850392d5e308cd6ea951c3700539883c28df1",
            "61481bd3be3a445d5a4b9070e8f8b2c6cc4fbbbbdc9f0e7ed048a132b8b84e0d",
            "e44c0d5cfcc6ad283011602a738fa28dfa1ad7f7540c9503205479072a9cc1ef",
            "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387",
        ),
        quantization="int8",
        languages=("multilingual",),
        language_mode=LanguageMode.OPTIONAL_HINT,
        required_sample_rate=16_000,
        preferred_chunk_seconds=15.0,
        max_chunk_seconds=30.0,
        supports_word_timestamps=False,
        expected_memory_mib=None,
        notes="Robustness candidate; CPU latency and memory are not yet admitted.",
    ),
)

_BY_ID = {profile.profile_id: profile for profile in _PROFILES}
_ALIASES = {"fast": DEFAULT_PROFILE_ID}


def list_asr_profiles() -> tuple[AsrModelProfile, ...]:
    return _PROFILES


def benchmark_profile_ids() -> tuple[str, ...]:
    return tuple(profile.profile_id for profile in _PROFILES)


def canonical_profile_id(profile_id: str) -> str:
    return _ALIASES.get(profile_id, profile_id)


def get_asr_profile(profile_id: str) -> AsrModelProfile:
    canonical = canonical_profile_id(profile_id)
    try:
        return _BY_ID[canonical]
    except KeyError as error:
        choices = ", ".join(_BY_ID)
        raise ValueError(f"unknown ASR profile {profile_id!r}; choose one of: {choices}") from error
