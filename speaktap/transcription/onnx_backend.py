"""Generic onnx-asr backend kept behind the SpeakTap ASR contract."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from ..domain import AsrCapabilities, AsrResult, AudioChunk
from ..profiles import AsrModelProfile


class OnnxAsrBackend:
    """Load one ONNX ASR model once and reuse it for all sessions."""

    def __init__(
        self,
        *,
        profile: AsrModelProfile,
        provider: str = "CPUExecutionProvider",
        threads: int = 0,
        inter_op_threads: int = 1,
        execution_mode: str = "parallel",
    ) -> None:
        if threads < 0 or inter_op_threads < 0:
            raise ValueError("thread counts must be non-negative")
        if execution_mode not in {"sequential", "parallel"}:
            raise ValueError("execution_mode must be sequential or parallel")
        self.backend_id = f"onnx-asr:{profile.profile_id}:{profile.model}:{profile.quantization}"
        self._profile = profile
        self._model_name = profile.model
        self._quantization = profile.quantization or None
        self._provider = provider
        self._threads = threads
        self._inter_op_threads = inter_op_threads
        self._execution_mode = execution_mode
        self._model: Any | None = None
        self._lock = threading.Lock()

    def capabilities(self) -> AsrCapabilities:
        return AsrCapabilities(
            required_sample_rate=self._profile.required_sample_rate,
            preferred_chunk_seconds=self._profile.preferred_chunk_seconds,
            max_chunk_seconds=self._profile.max_chunk_seconds,
            supports_word_timestamps=self._profile.supports_word_timestamps,
            supports_language_hint=self._profile.supports_language_hint,
            max_concurrency=1,
        )

    def load(self) -> None:
        # The check and the build are both held under the lock. Splitting them
        # would let two threads pass the check before either finished, and a
        # second build downloads, checksums, and holds an entire extra ONNX
        # session. The lock is uncontended in practice: one warm worker calls
        # this, and it is already re-acquired for inference immediately after.
        with self._lock:
            if self._model is None:
                self._model = self._build_model()

    def _build_model(self) -> Any:
        from huggingface_hub import snapshot_download
        from onnx_asr import load_model

        model_path = snapshot_download(
            repo_id=self._profile.repository,
            revision=self._profile.revision,
            allow_patterns=list(self._profile.files),
        )
        self._verify_model_files(Path(model_path))
        session_options = ort.SessionOptions()
        if self._threads:
            session_options.intra_op_num_threads = self._threads
        if self._inter_op_threads:
            session_options.inter_op_num_threads = self._inter_op_threads
        session_options.execution_mode = (
            ort.ExecutionMode.ORT_PARALLEL
            if self._execution_mode == "parallel"
            else ort.ExecutionMode.ORT_SEQUENTIAL
        )
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return load_model(
            self._model_name,
            path=model_path,
            quantization=self._quantization,
            sess_options=session_options,
            providers=[self._provider],
            preprocessor_config={
                "max_concurrent_workers": 1,
                "use_numpy_preprocessors": True,
            },
        )

    def _verify_model_files(self, model_path: Path) -> None:
        for filename, expected_digest in zip(
            self._profile.files,
            self._profile.sha256,
            strict=True,
        ):
            artifact = model_path / filename
            with artifact.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_digest:
                raise RuntimeError(
                    f"model artifact checksum mismatch for {self._profile.profile_id}: {filename}"
                )

    def warmup(self) -> None:
        self.load()
        # Half a second of silence at whatever rate the profile requires. Using
        # a fixed rate here would warm the graph with a resampling path the real
        # sessions never take.
        sample_rate = self._profile.required_sample_rate
        waveform = np.zeros(sample_rate // 2, dtype=np.float32)
        with self._lock:
            self._recognize(waveform, sample_rate=sample_rate, language="")

    def transcribe(self, chunk: AudioChunk, *, language: str = "") -> AsrResult:
        self.load()
        # frombuffer() initially returns a read-only view over the immutable bytes.
        # astype() intentionally makes a writable float32 copy: every operation
        # below mutates it in place, so replacing this with a zero-copy conversion
        # would make transcription fail.
        samples = np.frombuffer(chunk.pcm_s16le, dtype="<i2").astype(np.float32)
        samples *= 1.0 / 32768.0
        # Remove microphone DC bias, then give the decoder a consistent signal
        # level. The 0.95 target leaves headroom for rounding instead of placing
        # the loudest sample exactly on the clipping boundary.
        samples -= np.mean(samples)
        peak = float(np.max(np.abs(samples)))
        if peak > 1e-6:
            samples *= 0.95 / peak
        started = time.monotonic()
        with self._lock:
            text = self._recognize(
                samples,
                sample_rate=chunk.sample_rate,
                language=language,
            )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        return AsrResult(
            text=text.strip(),
            backend_id=self.backend_id,
            diagnostics={
                "duration_ms": elapsed_ms,
                "audio_ms": chunk.duration_ms,
                "realtime_factor": elapsed_ms / max(1, chunk.duration_ms),
            },
        )

    def close(self) -> None:
        self._model = None

    def _recognize(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        *,
        sample_rate: int,
        language: str,
    ) -> str:
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")
        if language:
            result = self._model.recognize(
                waveform,
                sample_rate=sample_rate,
                language=language,
            )
        else:
            result = self._model.recognize(waveform, sample_rate=sample_rate)
        if not isinstance(result, str):
            raise TypeError("onnx-asr returned a non-text result")
        return result
