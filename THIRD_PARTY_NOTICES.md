# Third-party notices

SpeakTap source code is Apache-2.0. Dependencies and downloaded model artifacts
remain under their own licenses.

No model weights, private recordings, benchmark corpora, or trained cleanup
checkpoints are distributed in this repository.

## Runtime software

- [`onnx-asr`](https://github.com/istupakov/onnx-asr) — MIT
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — MIT
- [NumPy](https://github.com/numpy/numpy) — BSD-3-Clause
- [Hugging Face Hub client](https://github.com/huggingface/huggingface_hub) —
  Apache-2.0

The exact resolved dependency versions and transitive packages are recorded in
`uv.lock`.

## ASR artifacts

- [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
  — CC-BY-4.0 according to its model repository; derived from NVIDIA Parakeet.
- [`istupakov/canary-1b-v2-onnx`](https://huggingface.co/istupakov/canary-1b-v2-onnx)
  — CC-BY-4.0 according to its model repository; derived from NVIDIA Canary.
- [`onnx-community/whisper-large-v3-turbo`](https://huggingface.co/onnx-community/whisper-large-v3-turbo)
  — converted from OpenAI Whisper (MIT). The conversion repository does not
  currently expose an explicit license in its metadata, so SpeakTap keeps it
  at candidate status and does not redistribute its artifacts.

Repository revisions and file hashes are pinned in `speaktap/profiles.py`.
Users downloading a model are responsible for reviewing and complying with its
license and acceptable-use terms.

## Research data

Cleanup research scripts can consume the external DISCO dataset. The inspected
upstream DISCO repository did not include an explicit license file, so derived
weights are local experiments and must not be redistributed through SpeakTap.
