# Model support

SpeakTap has an architecture-neutral ASR contract, but a model is called
supported only after CPU/ONNX compatibility, output quality, latency, memory,
language behavior, and long-session segmentation have been measured.

The installer selects one profile. Models are not switched while the service is
running.

| Profile | Architecture | Languages | Status |
| --- | --- | --- | --- |
| `parakeet-tdt-v3-int8` | TDT transducer | 25 European languages, automatic detection | Supported and default |
| `canary-1b-v2-int8` | Attention encoder-decoder | 25 European languages; explicit hint outside English | Benchmark candidate |
| `whisper-large-v3-turbo-int8` | Attention encoder-decoder | Multilingual; optional hint | Benchmark candidate |

The version numbers in these identifiers are upstream model names, not SpeakTap
release generations.

`fast` is an alias for the default profile, so `./install.sh --profile fast`
resolves to `parakeet-tdt-v3-int8`.

## Reproducible artifacts

Every profile pins:

1. a Hugging Face repository;
2. a full 40-character Git revision;
3. the exact files allowed for download;
4. a SHA-256 digest for every downloaded file.

SpeakTap validates those digests before loading ONNX Runtime. The authoritative
values live in `speaktap/profiles.py`; this avoids duplicating long checksum
tables in documentation.

| Profile | Repository | Revision |
| --- | --- | --- |
| Parakeet | `istupakov/parakeet-tdt-0.6b-v3-onnx` | `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce` |
| Canary | `istupakov/canary-1b-v2-onnx` | `5ebc1520cef7b6b318b3526ad17adbfe00bc1bfc` |
| Whisper | `onnx-community/whisper-large-v3-turbo` | `360ebcde2559d60bb474678be3c1de9ef347d01a` |

## Trying candidates

List profiles:

```bash
./install.sh --list-profiles
```

Benchmark candidates sequentially so only one model occupies memory:

```bash
uv run speaktap service-stop
uv run speaktap-model-benchmark /path/to/dataset
```

Installing a candidate is explicit:

```bash
./install.sh --profile canary-1b-v2-int8
```

Candidate status means the adapter loads, not that the model is recommended for
daily dictation. Parakeet is the only admitted profile at this release.

## Architecture policy

- Runtime artifacts must execute through ONNX Runtime on CPU.
- New architectures need a dedicated adapter when their tokenizer, decoder, or
  language controls differ.
- Streaming is implemented around bounded complete chunks; the model itself
  does not need native streaming.
- Unsupported capabilities must be declared in the profile rather than guessed
  at runtime.
- Promotion requires public multilingual benchmarks and a clear redistribution
  story.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for model licenses.
