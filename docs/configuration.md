# Configuration

SpeakTap reads `SPEAKTAP_*` environment variables when the service starts. The
model profile is the exception: `./install.sh --profile …` writes it to
`~/.config/speaktap/config.json`, and the running service deliberately ignores
`SPEAKTAP_PROFILE`. This prevents accidental model changes and multiple large
models in memory.

## Common settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPEAKTAP_LANGUAGE` | automatic | Language hint; required by some candidate profiles |
| `SPEAKTAP_AUDIO_DEVICE` | `default` | ALSA capture device |
| `SPEAKTAP_OUTPUTS` | `typing,clipboard,notification` | Comma-separated output adapters |
| `SPEAKTAP_SOUNDS_ENABLED` | `true` | Play start and stop feedback |
| `SPEAKTAP_START_SOUND` | freedesktop login sound | Start sound file |
| `SPEAKTAP_STOP_SOUND` | freedesktop logout sound | Stop sound file |
| `SPEAKTAP_CLEANUP_ENABLED` | `true` | Enable deterministic safe cleanup |
| `SPEAKTAP_MAX_RECORDING_SECONDS` | `300` | Hard recording limit |

Example:

```bash
SPEAKTAP_LANGUAGE=fr \
SPEAKTAP_OUTPUTS=typing,clipboard,notification \
./toggle.sh
```

Desktop shortcuts do not always inherit shell profiles. Put variables directly
in the shortcut command or call a wrapper script when you need persistent
overrides.

## Audio and segmentation

| Variable | Default |
| --- | ---: |
| `SPEAKTAP_SAMPLE_RATE` | `16000` |
| `SPEAKTAP_FRAME_MILLISECONDS` | `20` |
| `SPEAKTAP_SPEECH_PADDING_MILLISECONDS` | `500` |
| `SPEAKTAP_MIN_CHUNK_SECONDS` | `4` |
| `SPEAKTAP_TARGET_CHUNK_SECONDS` | `15` |
| `SPEAKTAP_MAX_CHUNK_SECONDS` | `25` |
| `SPEAKTAP_SILENCE_MILLISECONDS` | `600` |
| `SPEAKTAP_FORCED_CUT_OVERLAP_MILLISECONDS` | `400` |
| `SPEAKTAP_MAX_PENDING_CHUNKS` | `8` |

The defaults are tuned as one set. A profile may impose a stricter sample rate
or maximum chunk duration, in which case invalid values fail at startup.

## Runtime tuning

| Variable | Default | Notes |
| --- | --- | --- |
| `SPEAKTAP_ONNX_EXECUTION_PROVIDER` | `CPUExecutionProvider` | CPU is the supported provider |
| `SPEAKTAP_THREADS` | `0` | ONNX Runtime chooses the intra-op count |
| `SPEAKTAP_INTER_OP_THREADS` | `1` | Inter-op thread count |
| `SPEAKTAP_EXECUTION_MODE` | `parallel` | `parallel` or `sequential` |
| `SPEAKTAP_SPEECH_DETECTOR` | `adaptive_energy` | Current detector adapter |

Advanced ASR artifact variables exist for controlled development, but a valid
runtime must match the installed profile exactly:
`SPEAKTAP_ASR_ADAPTER`, `SPEAKTAP_ASR_MODEL`, and
`SPEAKTAP_ASR_QUANTIZATION`.

Cleanup development variables are `SPEAKTAP_CLEANUP_ADAPTER` (default `safe`),
`SPEAKTAP_CLEANUP_MODEL` (empty), and
`SPEAKTAP_CLEANUP_TIMEOUT_SECONDS` (default `2`).

## Files

- Installed profile: `~/.config/speaktap/config.json`
- Session diagnostics: `~/.local/state/speaktap/`
- Runtime socket and PID: a private, user-owned directory below
  `${XDG_RUNTIME_DIR:-/tmp}`
- Downloaded models: the Hugging Face cache, normally
  `~/.cache/huggingface/hub/`

Stop the service before changing runtime settings:

```bash
uv run speaktap service-stop
```
