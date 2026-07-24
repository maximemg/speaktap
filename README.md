# SpeakTap

Fast, private, local dictation for Linux.

Press a keyboard shortcut to start recording, press it again to stop, and
SpeakTap types the transcript into the focused application. Audio and text stay
on your machine.

SpeakTap is built for long-form dictation on CPU. It keeps one ONNX model warm,
transcribes completed speech chunks while you continue talking, removes only
unambiguous filler/repetition noise, and falls back safely when an optional
desktop integration fails.

## What works today

- X11 desktop sessions on Ubuntu and Debian-based distributions
- CPU inference through ONNX Runtime
- English, French, and 23 other European languages with the default model
- Toggle-to-record keyboard shortcuts
- Direct typing, clipboard copy, desktop notifications, and start/stop sounds
- Short notes and multi-minute dictation through the same streaming pipeline
- Deterministic cleanup of explicit fillers and immediate repetitions
- Reproducible ASR and cleanup benchmarks

The default profile is NVIDIA Parakeet TDT 0.6B v3 INT8. SpeakTap downloads a
pinned ONNX revision during installation; model files are not stored in this
repository.

## Requirements

- Linux x86-64
- An X11 session
- A working microphone exposed through ALSA
- Around 2 GiB of free RAM for the default model
- [`uv`](https://docs.astral.sh/uv/) for the managed Python environment

On Ubuntu, install the desktop helpers with:

```bash
sudo apt install alsa-utils xdotool xclip libnotify-bin pulseaudio-utils
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

```bash
git clone https://github.com/maximemg/speaktap.git
cd speaktap
./install.sh
```

The first installation downloads and warms the default ASR model. Later
commands reuse the local model cache and the long-lived service.

Test the interaction:

```bash
./toggle.sh  # start recording
./toggle.sh  # stop, transcribe, type, copy, and notify
```

## Add a keyboard shortcut

On Ubuntu:

1. Open **Settings → Keyboard → View and Customize Shortcuts → Custom
   Shortcuts**.
2. Add a shortcut named `SpeakTap`.
3. Use the absolute path to `toggle.sh` as the command.
4. Assign any convenient key combination.

Example command:

```text
/home/you/Projects/speaktap/toggle.sh
```

The same shortcut starts and stops recording.

## Useful commands

```bash
uv run speaktap status
uv run speaktap start
uv run speaktap stop
uv run speaktap cancel
uv run speaktap service-stop
uv run speaktap-transcribe-file /path/to/16khz-mono-pcm16.wav
uv run speaktap-models list
```

The installed model is selected during installation, not switched while the
service is running:

```bash
./install.sh --list-profiles
./install.sh --profile parakeet-tdt-v3-int8
```

Only Parakeet is currently recommended. Canary and Whisper profiles remain
benchmark candidates and may require substantially more CPU time and memory.

## Configuration

SpeakTap uses `SPEAKTAP_*` environment variables. Common examples:

```bash
export SPEAKTAP_LANGUAGE=fr
export SPEAKTAP_AUDIO_DEVICE=default
export SPEAKTAP_SOUNDS_ENABLED=true
export SPEAKTAP_CLEANUP_ENABLED=true
export SPEAKTAP_OUTPUTS=typing,clipboard,notification
```

If these must apply to a desktop shortcut, place them directly in the shortcut
command or in a small wrapper script. See
[configuration](docs/configuration.md) for every setting.

## Privacy and safety

- Microphone audio is processed locally.
- SpeakTap does not upload recordings or transcripts.
- Models are downloaded from their pinned Hugging Face revisions.
- Session diagnostics contain text and timing metadata under
  `~/.local/state/speaktap/`; protect or delete them if the dictated content is
  sensitive.
- Cleanup never runs commands or answers dictated questions. If cleanup fails,
  SpeakTap emits the assembled raw transcript.

## Current limitations and roadmap

- **Wayland is not supported yet.** Native `wtype`/`wl-copy` output is the first
  desktop portability milestone.
- Installation has been exercised primarily on Ubuntu 24.04 under X11.
- CPU is the only supported execution target; GPU providers are intentionally
  deferred.
- The adaptive energy detector can still struggle with rapidly changing
  background noise. An ONNX neural VAD adapter is planned.
- Parakeet remains the only admitted ASR profile.
- Learned text cleanup is not enabled: every evaluated multilingual candidate
  was less safe or less accurate than deterministic cleanup.
- Distribution packages for Debian, Fedora, Arch, and PyPI are not provided
  yet.

Contributions around Wayland, additional Linux distributions, reproducible
model admission, and accessibility are welcome.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Models and licenses](docs/models.md)
- [Benchmarking](docs/benchmarks.md)
- [Cleanup-model research](docs/cleanup-model-research.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

SpeakTap is licensed under the [Apache License 2.0](LICENSE). Downloaded model
artifacts retain their own licenses and attribution requirements; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
