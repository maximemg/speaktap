# Cleanup candidate benchmark

This text-only benchmark compares optional post-ASR cleanup architectures
without storing or publishing anyone's voice. Cases cover English and French
disfluencies, punctuation, false starts, protected technical text, questions
and commands, and already-clean passthrough.

Each candidate runs in a separate process. Reports include exact match, content
word error, character error, protected-anchor preservation, forbidden answer
generation, warm latency, load time, and resident memory.

The benchmark is deliberately stricter than ordinary grammar correction:

- dictated questions must be cleaned rather than answered;
- names, numbers, URLs, and code must not be silently changed;
- unsupported languages must pass through safely;
- timeout or validation failure must fall back to the assembled ASR text.

Candidate dependencies and model artifacts are not part of the normal
installation. The runner is an admission tool, not a production adapter.

The `qwen` adapter uses the CPU INT4 ORT GenAI artifact from
`onnx-community/Qwen3-0.6B-ONNX`. `--prompt-profile` selects the reproducible
`fewshot`, `strict`, `concise`, `conservative`, or `strict-thinking` prompt;
`--num-beams` selects greedy or beam decoding. Non-thinking greedy decoding
remains the normal comparison because it is deterministic and avoids loading
PyTorch or Optimum.

The `qwen-constrained` and `qwen-constrained-delete` adapters ask the same model
for original token indexes rather than rewritten text. Their deterministic
validator permits only protected, structurally safe selections and falls back
to the raw transcript on malformed or unsafe output. The `deterministic`
adapter is the equivalent model-free filler/repetition baseline.

## Multilingual edit-tagger experiment

`train_multilingual_tagger.py` fine-tunes
`distilbert-base-multilingual-cased` as a deletion-only KEEP/DELETE classifier
using DISCO's English, French, German, and Hindi workbooks. It selects a
false-deletion-constrained threshold from held-out validation data, evaluates
an untouched test split, exports dynamic ONNX, quantizes it to INT8, validates
the graph, and runs an inference probe.

```bash
git clone --depth 1 https://github.com/vineet2104/DISCO.git /tmp/disco

uv run --script train_multilingual_tagger.py \
  --data-dir /tmp/disco/data/labeled-data \
  --output /tmp/multilingual-edit-tagger \
  --device cpu

uv run --project ../.. --with 'tokenizers>=0.22,<0.23' \
  python run_candidate.py \
  --adapter multilingual-tagger \
  --model-root /tmp/multilingual-edit-tagger \
  --output ../../benchmark/results/cleanup-multilingual-tagger.json

uv run --script evaluate_multilingual_tagger.py \
  --data-dir /tmp/disco/data/labeled-data \
  --model-root /tmp/multilingual-edit-tagger \
  --output ../../benchmark/results/cleanup-multilingual-tagger-corpus.json
```

The trainer refuses data whose DELETE-label density falls outside 10-35%.
This guards against accidental source/source alignment. The upstream DISCO
repository currently has no explicit `LICENSE` file, so trained artifacts are
local experiments and must not be redistributed until licensing is clarified.

The 2026-07-24 checkpoint is not admitted: safe thresholding preserved all
protected benchmark anchors and ran in 10 ms median, but its 16.8% content WER
lost to deterministic cleanup at 15.2%. See
`../../docs/cleanup-model-research.md`.

The model output directory is deliberately outside the repository. The normal
installer does not download it, and the production configuration remains on
`SafeCleaner`. Keep the scripts and adapter as reproducible evaluation
infrastructure; do not package the checkpoint.
