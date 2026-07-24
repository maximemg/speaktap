# Cleanup candidate results — 2026-07-24

## Decision

No evaluated model is admitted as the complete cleanup stage.

- Use `SafeCleaner` as the default and retain `NoopCleaner` as the raw-output
  escape hatch.
- Keep the ASR profile on Parakeet TDT v3/int8.
- Retain encoder-only punctuation tagging as a viable optional component.
- Reject Qwen3 0.6B for cleanup in both raw-generation and constrained-edit
  modes; the constrained version is safe but loses to deterministic rules.
- Retain deterministic filler and repetition removal as the safe cleanup
  baseline.
- Reject the trained multilingual Distil-mBERT edit tagger as the default: its
  safely calibrated threshold is fast and preserves protected anchors, but its
  content WER is worse than deterministic cleanup.
- Reject the evaluated Typurr and Mumble profiles.
- Do not pursue the evaluated multilingual mT5 checkpoint.
- Require multilingual evidence, not English-only or bilingual evidence, before
  enabling learned cleanup by default.

### Final disposition

- **Production:** keep `SafeCleaner` enabled and `NoopCleaner` available. No
  cleanup-model install, runtime switch, or user reinstall is required.
- **Keep in the repository:** the architecture-neutral benchmark runner,
  deterministic safety checks, multilingual trainer, ONNX exporter, corpus
  evaluator, and regression tests. They make the next candidate directly
  comparable instead of restarting the investigation.
- **Do not ship:** the rejected model weights or their training-only
  dependencies. Candidate packages remain outside the normal installation.
- **Do not delete the experiment:** the deletion-tagger path proved that a
  multilingual encoder can run quickly on CPU through ONNX Runtime. The trained
  checkpoint, not the adapter architecture, is rejected.
- **Do not tune this checkpoint further:** deletion-only inference cannot add
  punctuation, replace a mistaken word, or resolve a correction by rewriting.
  More epochs would not remove that architectural limit.

## Method

The public text-only set in `cases.jsonl` contains 16 English/French cases:
fillers, repeated words, false starts, punctuation, time formatting, protected
technical text, dictated questions and commands, and clean passthrough.
No microphone recordings or private voice data are used.

Each viable candidate ran in a fresh process on CPU after one unmeasured
warm-up. The test machine was an Intel Core i7-12700H with 20 logical CPUs.
Metrics are:

- content WER after case-folding and removing punctuation;
- exact output match, including punctuation and capitalization;
- exact preservation of protected names, numbers, URLs, and code;
- forbidden answer/refusal text emitted for dictated content;
- warm per-case latency and process RSS.

The 16-case suite is an admission regression set, not a statistically complete
measure of conversational speech. Its purpose is to expose unsafe changes and
compare candidates under identical conditions. The larger DISCO-derived test
split measures token deletion separately; it does not replace end-to-end ASR
evaluation because its targets are written disfluency edits rather than
microphone transcripts.

## Results

| Profile | Architecture | Artifact | Exact | Content WER | Protected | Median | P90 | RSS | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `typurr-edit-tagger` | Encoder-only multi-head edit tagger | 312 MiB ONNX fp32 | 12.5% | 12.8% | 53.3% | 44 ms | 53 ms | 501 MiB | Rejected |
| `punctuate-all-int8` | Encoder-only punctuation tagger | 265 MiB ONNX int8 | 25.0% | 20.8% | 100.0% | 8 ms | 11 ms | 709 MiB | Partial component only |
| `deterministic-disfluency` | Validated rules | None | 12.5% | 15.2% | 100.0% | <1 ms | <1 ms | 35 MiB | Safe baseline |
| `multilingual-edit-tagger` | Distil-mBERT KEEP/DELETE tagger | 129 MiB ONNX int8 | 12.5% | 16.8% | 100.0% | 10 ms | 17 ms | 253 MiB | Safe threshold rejected |
| `qwen3-0.6b-int4` | Decoder-only generative model | 512 MiB ONNX/ORT GenAI int4 | 12.5% | 11.2% | 93.3% | 1787 ms | 2247 ms | 961 MiB | Candidate rejected in raw form |
| `qwen3-token-delete` | Validated generative token selection | 512 MiB ONNX/ORT GenAI int4 | 12.5% | 20.8% | 100.0% | 1397 ms | 1623 ms | 1019 MiB | Safe but worse than rules |
| `mumble-cleanup-int8` | Decoder-only Qwen 0.5B | 473 MiB ONNX int8 | 12.5% | 110.4% | 40.0% | 2989 ms | 5744 ms | 2284 MiB | Rejected |
| multilingual mT5-small | Encoder-decoder | 1.2 GiB framework weights | — | — | — | >20 s observed | — | — | Export-blocked and rejected |
| custom multilingual edit tagger | Encoder-only edit tagger | Trained locally | 12.5% | 16.8% | 100.0% | 10 ms | 17 ms | 253 MiB | First experiment rejected |

### Per-language content results

| Profile | English WER | English protected | French WER | French protected |
| --- | ---: | ---: | ---: | ---: |
| `typurr-edit-tagger` | 10.8% | 62.5% | 15.0% | 42.9% |
| `punctuate-all-int8` | 18.5% | 100.0% | 23.3% | 100.0% |
| `deterministic-disfluency` | 13.8% | 100.0% | 16.7% | 100.0% |
| `multilingual-edit-tagger` | 13.8% | 100.0% | 20.0% | 100.0% |
| `qwen3-0.6b-int4` | 6.2% | 100.0% | 16.7% | 85.7% |
| `qwen3-token-delete` | 18.5% | 100.0% | 23.3% | 100.0% |
| `mumble-cleanup-int8` | 47.7% | 62.5% | 178.3% | 14.3% |

The punctuation tagger's content WER reflects its intentionally limited scope:
it preserves fillers and false starts rather than deleting them.

### Trained multilingual edit tagger

- Fine-tuned `distilbert-base-multilingual-cased` as a deletion-only token
  classifier on 13,165 usable DISCO pairs across English, French, German, and
  Hindi.
- The data pipeline derives KEEP/DELETE labels by aligning the fluent target as
  the rightmost word subsequence of the disfluent source. It rejects 95 pairs
  that cannot be represented as deletion-only edits.
- A label-density assertion verified 21,361 DELETE labels among 116,223 source
  tokens (18.38%) before training. This check caught and invalidated an initial
  header-matching bug before that artifact reached the benchmark.
- The best three-epoch model reached 0.905 deletion F1 at threshold 0.5 on the
  held-out validation split.
- Calibration selected threshold 0.99 to satisfy the validation false-deletion
  ceiling. Before quantization, the independent corpus test split reached
  99.61% deletion precision, 58.24% recall, 0.735 F1, and a 0.054%
  false-delete rate.
- The exported dynamic INT8 ONNX artifact is 129 MiB. With deterministic
  bypasses for code/path dictation, it preserved every protected benchmark
  anchor and produced no forbidden answer or command-execution text.
- At the safe 0.99 threshold it scored 16.8% content WER, worse than the 15.2%
  deterministic baseline. At diagnostic threshold 0.5 it reached 13.6% WER,
  but corpus precision fell to 92.67% with a 1.59% false-delete rate.
- Running `SafeCleaner` before the learned tagger did not improve WER at safe
  thresholds; the tagger made no additional accepted corrections.
- The experiment therefore proves the runtime architecture is viable and fast,
  but this checkpoint is not accurate enough to enable. The reproducible
  trainer remains in `train_multilingual_tagger.py`; the artifact must not be
  redistributed because the upstream DISCO repository has no explicit license
  file.

The exported INT8 model was evaluated again rather than assuming framework and
ONNX parity:

| Test scope | Examples | Delete precision | Delete recall | Delete F1 | False-delete rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| All languages | 1,318 | 99.58% | 54.64% | 0.706 | 0.054% |
| English | 398 | 100.00% | 74.21% | 0.852 | 0.000% |
| French | 298 | 99.57% | 54.74% | 0.706 | 0.058% |
| German | 306 | 98.69% | 65.23% | 0.785 | 0.229% |
| Hindi | 316 | 100.00% | 27.93% | 0.437 | 0.000% |

Quantization preserved precision but reduced aggregate deletion recall from
58.24% to 54.64%. Hindi recall is particularly weak, so the checkpoint cannot
be described as uniformly multilingual even though its base encoder and
training corpus cover all four languages.

Threshold experiments exposed the core trade-off:

| Delete threshold | 16-case WER | Protected anchors | Safety interpretation |
| ---: | ---: | ---: | --- |
| 0.99 | 16.8% | 100.0% | Safely calibrated but worse than rules |
| 0.95 | 15.2% | 100.0% | Ties rules; lower validation precision adds risk without value |
| 0.50 | 13.6% | 100.0% | Better small-suite WER, but only 92.67% deletion precision and 1.59% false deletes |

Protected-anchor checks cannot prove that ordinary words were preserved. The
0.50 result is therefore not admissible despite its lower WER on the small
suite: corpus evaluation demonstrates that it deletes valid, unmarked content.
Likewise, cascading `SafeCleaner` into the model at thresholds 0.95, 0.975, or
0.99 remained at 15.2% WER, so the model added no safe improvement.

### What the experiments establish

1. Deterministic rules currently provide the best complete safety/performance
   trade-off: 15.2% WER, 100% protected preservation, sub-millisecond latency,
   and no model memory.
2. Encoder taggers are fast enough for this pipeline, but safety must cover all
   ordinary content—not only identifiers, URLs, names, and numbers.
3. A punctuation-only tagger is technically viable, but Parakeet already emits
   punctuation and the measured candidate does not remove disfluencies.
4. Small generative models can improve aggregate WER while silently changing
   meaning. Prompting, beam search, thinking, and constrained index generation
   did not produce a safe win.
5. A future learned cleaner must be multilingual by measured per-language
   behavior, distributable under a clear license, ONNX-compatible, and capable
   of either validated edits or exact raw-text fallback.
6. Cleanup is optional. It must never compensate for a weaker ASR profile or
   become a prerequisite for reliable dictation.

## Important failures

### Typurr

- Changed “three thirty” to “three.”
- Changed exact identifiers such as `process_audio_chunk` and
  `api.example.com/v2` through unwanted capitalization.
- Deleted content from dictated commands.
- Applied erratic capitalization throughout French.
- The model repository does not publish the output label map or reconstruction
  code; the ONNX graph exposes undocumented `keep`, `punct`, and `cap` heads.

Its latency is excellent, but the preservation failures make it unsafe.

### Punctuate-all

- Preserved every protected anchor.
- Ran in 8 ms median after dynamic int8 quantization.
- Restored useful punctuation in both English and French.
- Did not remove fillers, repetitions, or self-corrections.
- Sometimes chose a period where a question mark was expected.

This is the only candidate worth retaining, but only as a punctuation adapter,
not as the complete cleanup stage. Parakeet already emits punctuation, so its
incremental product value must be measured before integration.

### Qwen3 0.6B

- Used the CPU-specific INT4 artifact through ONNX Runtime GenAI 0.14.1 on the
  project's Python 3.14 runtime.
- The initial zero-shot prompt was safe but mostly copied its input: 19.2%
  content WER, 93.3% anchor preservation, and no forbidden answers.
- Three bilingual editing examples improved content WER to 11.2% without
  producing an answer, refusal, or executed command.
- Preserved all English anchors and every technical identifier in both
  languages.
- Silently dropped meaningful French words including “vendredi,” “demain,”
  and “maintenant”; this is a faithfulness failure that a simple protected-span
  check cannot reliably detect.
- Took 1.8 seconds median per short input and approximately 961 MiB RSS. This
  is much better than Mumble but still material beside the ASR model on CPU.

Prompt and decoding settings were then tested explicitly:

| Prompt | Search | Exact | Content WER | Protected | Median | P90 | RSS | Important result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Zero-shot | Greedy, no thinking | 12.5% | 19.2% | 93.3% | 1490 ms | 2024 ms | 925 MiB | Safe but mostly copied input |
| Bilingual few-shot | Greedy, no thinking | 12.5% | **11.2%** | 93.3% | 1787 ms | 2247 ms | 961 MiB | Best WER; deleted French content |
| Strict minimal-edit | Greedy, no thinking | **31.3%** | 15.2% | 93.3% | 2013 ms | 2175 ms | 1015 MiB | Preserved French endings but chose Thursday instead of corrected Friday |
| Concise rules | Greedy, no thinking | 18.8% | 12.0% | 93.3% | **1420 ms** | **1611 ms** | 958 MiB | Faster; made the same wrong Thursday choice |
| Concise rules | Two beams, no thinking | 18.8% | 12.8% | 93.3% | 2516 ms | 2823 ms | 1303 MiB | Slower and less accurate |
| Conservative, no correction | Greedy, no thinking | 31.3% | 19.2% | 93.3% | 1416 ms | 1682 ms | 949 MiB | Violated the rule, deleted French content, and translated technical text |
| Strict minimal-edit | Greedy, thinking | 0.0% | Invalid | 66.7% | 8249 ms | 9738 ms | 1043 MiB | Repetitive reasoning often exhausted the 256-token budget |

All non-thinking profiles produced zero forbidden answers or command
executions. Sampling was not tested because it would make a safety-sensitive
cleanup stage nondeterministic, and there is no reliable scorer for choosing
the semantically correct sample.

Instructions clearly affect Qwen, but no prompt fixed the central faithfulness
problem. The best-WER prompt silently deleted content; the stricter prompts
either left disfluencies untouched or selected the wrong side of a correction.
Even a prompt explicitly forbidding correction resolution was ignored on
French input. Beam search and thinking did not help.

#### Constrained edit experiment

Two constrained protocols made model-generated text impossible:

1. Qwen selected the ordered indexes of original tokens to keep.
2. Qwen selected only indexes to delete.

Deterministic code parsed the exact JSON schema, reconstructed output only from
original tokens, protected identifiers and final content, limited ordinary
deletions to known fillers or adjacent repetitions, and allowed correction
edits only as validated prefix/suffix splices. Invalid output fell back to the
raw transcript.

| Mode | Content WER | Protected | Fallback | Median | P90 | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Token keep-list | 20.8% | 100.0% | 25.0% | 2338 ms | 2827 ms | 1023 MiB |
| Token delete-list, fully validated | 20.8% | 100.0% | 25.0% | 1397 ms | 1623 ms | 1019 MiB |
| Deterministic filler/repetition rules | **15.2%** | 100.0% | 0.0% | **<1 ms** | **<1 ms** | **35 MiB** |

The keep-list model mostly selected every token. The shorter delete-list model
mostly returned an empty list. On the one English correction it attempted,
Qwen copied an example's index pattern and produced “send the it on Friday.”
The splice validator rejected that result and returned the raw input. Qwen also
wrapped three JSON responses in Markdown despite explicit output instructions.

Constrained editing solves the hallucination problem but exposes the more basic
issue: Qwen3 0.6B is not identifying disfluencies reliably. It is slower, uses
nearly 1 GiB more memory, and has 5.6 absolute points worse WER than the
deterministic baseline. Qwen3 0.6B should not be integrated for cleanup.

### Mumble

- Changed `HTTP 429` to `HTTP/1.1`.
- Answered a dictated question with “The capital of France is Paris.”
- Turned spoken command text into `rm -rf README.md`.
- Replaced a clean sentence with meta-commentary.
- Produced English refusals and instructions for French input.
- Required an Optimum/PyTorch dependency path despite using ONNX and consumed
  more than 2 GiB RSS in that benchmark process.

These are hard safety and faithfulness failures, not tuning issues.

### Multilingual mT5-small

- Failed both cached and non-cached ONNX export with the current
  Python 3.14/Optimum stack.
- Required its historical Transformers version to load its tokenizer.
- Duplicated the first English probe sentence.
- Mostly passed other text through without punctuation or cleanup.
- Took more than 80 seconds for four short framework probes.

There is no reason to repair the exporter for this checkpoint.

## Admission requirements for the next candidate

A future cleanup profile must:

1. support English and French or pass unsupported text through byte-faithfully;
2. preserve names, numbers, URLs, commands, and code identifiers;
3. clean dictated questions and commands without answering or executing them;
4. beat deterministic cleanup on disfluency precision and recall;
5. stay comfortably below the ASR post-stop budget on CPU;
6. ship or reproducibly export to ONNX Runtime;
7. fall back to the assembled raw transcript on timeout or validation failure.
