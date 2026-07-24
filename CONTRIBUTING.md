# Contributing

Thanks for helping improve SpeakTap.

## Development setup

```bash
git clone https://github.com/maximemg/speaktap.git
cd speaktap
uv sync --locked --all-groups
```

Run the same checks as CI:

```bash
uv run ruff check .
uv run mypy speaktap tests
uv run pytest
uv build
```

Shell changes should also pass:

```bash
shellcheck install.sh toggle.sh
```

## Pull requests

- Keep microphone recordings, transcripts, model weights, and generated
  benchmark outputs out of commits.
- Add tests for behavior changes.
- Update user documentation when a command, setting, model status, or platform
  requirement changes.
- Benchmark model and segmentation changes on public data.
- Do not promote a candidate model without CPU/ONNX, multilingual, license,
  accuracy, latency, memory, and long-session evidence.

By contributing, you agree that your contribution is licensed under
Apache-2.0.
