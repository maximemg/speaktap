#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=""
LIST_PROFILES=false

usage() {
    echo "Usage: $0 [--profile PROFILE] [--list-profiles]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            if [ "$#" -lt 2 ]; then
                echo "--profile requires a value" >&2
                exit 2
            fi
            PROFILE="$2"
            shift 2
            ;;
        --list-profiles)
            LIST_PROFILES=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

missing=()
for command_name in arecord xdotool xclip notify-send paplay; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        missing+=("$command_name")
    fi
done
if [ "${#missing[@]}" -ne 0 ]; then
    echo "Missing Linux helpers: ${missing[*]}" >&2
    echo "On Ubuntu, install: alsa-utils xdotool xclip libnotify-bin pulseaudio-utils" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv sync --project "$SCRIPT_DIR" --locked

if [ "$LIST_PROFILES" = true ]; then
    uv run --project "$SCRIPT_DIR" speaktap-models list
    exit 0
fi

if [ -z "$PROFILE" ]; then
    PROFILE="$(uv run --project "$SCRIPT_DIR" speaktap-models current --id-only)"
    if [ -t 0 ]; then
        echo
        echo "Available CPU/ONNX ASR profiles:"
        uv run --project "$SCRIPT_DIR" speaktap-models list
        echo
        read -r -p "ASR profile [$PROFILE]: " selected_profile
        if [ -n "$selected_profile" ]; then
            PROFILE="$selected_profile"
        fi
    fi
fi

PROFILE="$(uv run --project "$SCRIPT_DIR" speaktap-models check "$PROFILE" --id-only)"
uv run --project "$SCRIPT_DIR" speaktap service-stop >/dev/null || true

uv run --project "$SCRIPT_DIR" python - "$PROFILE" <<'PY'
import sys

from speaktap.config import SpeakTapConfig
from speaktap.transcription import OnnxAsrBackend

config = SpeakTapConfig.from_env({"SPEAKTAP_PROFILE": sys.argv[1]})
print(
    f"Preparing {config.model_profile.display_name} "
    f"[{config.model_profile.status.value}]..."
)
backend = OnnxAsrBackend(
    profile=config.model_profile,
    provider=config.execution_provider,
    threads=config.asr_threads,
    inter_op_threads=config.asr_inter_op_threads,
    execution_mode=config.asr_execution_mode,
)
backend.load()
backend.warmup()
print(f"Dictation model is ready: {config.asr_model} ({config.asr_quantization})")
PY

uv run --project "$SCRIPT_DIR" speaktap-models select "$PROFILE"

echo
echo "Dictation is ready."
echo "Test toggle: $SCRIPT_DIR/toggle.sh"
echo "Safe file test: uv run --project $SCRIPT_DIR speaktap-transcribe-file AUDIO.wav"
