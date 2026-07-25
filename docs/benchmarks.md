# Benchmarking

SpeakTap includes two CPU/ONNX benchmark commands:

- `speaktap-benchmark` measures one profile on standalone clips and synthetic
  real-time long sessions.
- `speaktap-model-benchmark` runs several profiles sequentially, in isolated
  processes, so large models are never resident together.

Cleanup is disabled for raw ASR measurements.

## Dataset format

A dataset directory contains 16 kHz, mono, signed 16-bit little-endian PCM
files plus an exact transcript map:

```text
dataset/
|-- metadata.env
|-- reference.txt
|-- normal-01.raw
`-- noisy-01.raw
```

`metadata.env` may define one language:

```text
DATASET_LANGUAGE=en
```

For a mixed-language directory, omit that file and prefix each filename with a
language code, such as `en-normal-01.raw` and `fr-normal-01.raw`.

Each line in `reference.txt` maps the filename stem to the words spoken:

```text
normal-01 This is the exact dictated sentence.
noisy-01 The background fan should not remove these words.
```

Create compatible audio with:

```bash
arecord -q -f S16_LE -r 16000 -c 1 -t raw dataset/normal-01.raw
```

Personal recordings are useful locally but must not become release evidence or
be committed. Public corpora such as LibriSpeech and FLEURS make published
results reproducible without leaking a contributor's voice.

## Benchmark one profile

Stop daily dictation first so its warm model does not distort memory results:

```bash
uv run speaktap service-stop
uv run speaktap-benchmark \
  /path/to/librispeech-dataset \
  /path/to/fleurs-dataset \
  --profile parakeet-tdt-v3-int8 \
  --limit 20 \
  --passes 1 \
  --long-seconds 30,60 \
  --results-dir benchmark/results/run
```

The output directory contains per-clip CSV, `summary.json`, `REPORT.md`, and
long-session transcripts. It is ignored by Git because it may contain derived
audio and private text.

## Compare profiles

```bash
uv run speaktap-model-benchmark \
  /path/to/librispeech-dataset \
  /path/to/fleurs-dataset \
  --profiles parakeet-tdt-v3-int8,canary-1b-v2-int8,whisper-large-v3-turbo-int8 \
  --limit 20 \
  --passes 1 \
  --long-seconds 30,60
```

Each profile runs in a fresh child process. A failed candidate is recorded and
the suite continues. The installed daily-use profile is not changed.

## Development snapshot

The latest retained public-data snapshot was measured on July 23, 2026 on one
CPU machine with one measured pass after warm-up:

| Workload | Raw WER | Post-stop latency |
| --- | ---: | ---: |
| LibriSpeech, 20 clips / 193.6 s | 3.50% | Standalone throughput run |
| French FLEURS, 20 clips / 211.9 s | 7.72% | Standalone throughput run |
| 31.7 s synthetic real-time dictation | 2.94% | 908 ms |
| 60.0 s synthetic real-time dictation | 2.05% | 1442 ms |

Warm benchmark-process RSS was approximately 1233 MiB. The 60-second session
used 11 natural silence boundaries, no forced cut, and no ASR queue wait.

This is regression evidence, not a universal quality claim. Microphone,
acoustics, accents, names, numbers, noise, and the full five-minute recording
limit require broader evaluation.

## Admission policy

A candidate can become supported only when it:

1. runs through ONNX Runtime on CPU;
2. preserves or improves multilingual quality on public data;
3. stays within acceptable post-stop latency and resident memory;
4. completes long-session and repeated-start/stop reliability tests;
5. has pinned artifacts and clear licensing;
6. declares language and architecture limitations explicitly.

See [models](models.md) and [cleanup-model research](cleanup-model-research.md).
