"""Persistent structured diagnostics for successful and failed sessions."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import SpeakTapConfig
from .domain import TranscriptionResult


class SessionLogger:
    """Append one compact JSON object per completed session."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def success(
        self,
        result: TranscriptionResult,
        *,
        service_post_stop_ms: int,
        output_errors: tuple[str, ...],
    ) -> None:
        chunks = [
            {
                "sequence": chunk.sequence,
                "audio_start_ms": chunk.audio_start_ms,
                "audio_end_ms": chunk.audio_end_ms,
                "audio_ms": chunk.audio_end_ms - chunk.audio_start_ms,
                "speech_ms": chunk.speech_ms,
                "overlap_ms": chunk.overlap_ms,
                "cut_reason": chunk.cut_reason.value,
                "queue_wait_ms": chunk.queue_wait_ms,
                "asr_ms": chunk.asr_duration_ms,
                "text_chars": len(chunk.raw_text),
                "error": chunk.error,
            }
            for chunk in result.chunks
        ]
        audio_ms = max(
            (chunk.audio_end_ms for chunk in result.chunks),
            default=0,
        )
        self._append(
            {
                "schema_version": 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "pipeline_version": 3,
                "status": "ok",
                "session_id": result.session_id,
                "raw_text_chars": len(result.raw_text),
                "output_text_chars": len(result.output_text),
                "cleanup_status": result.cleanup_status.value,
                "audio_ms": audio_ms,
                "speech_ms": sum(chunk.speech_ms for chunk in result.chunks),
                "chunk_count": len(result.chunks),
                "timings_ms": {
                    **result.timings_ms,
                    "service_post_stop": service_post_stop_ms,
                },
                "memory": _process_memory(),
                "output_errors": list(output_errors),
                "chunks": chunks,
            }
        )

    def failure(
        self,
        *,
        session_id: str | None,
        error: BaseException | str,
        service_post_stop_ms: int,
    ) -> None:
        self._append(
            {
                "schema_version": 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "pipeline_version": 3,
                "status": "error",
                "session_id": session_id,
                "error": str(error),
                "timings_ms": {"service_post_stop": service_post_stop_ms},
                "memory": _process_memory(),
            }
        )

    def _append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as destination:
            fcntl.flock(destination.fileno(), fcntl.LOCK_EX)
            destination.write(payload)
            destination.write("\n")
            destination.flush()
            fcntl.flock(destination.fileno(), fcntl.LOCK_UN)


def _process_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines():
            key, _, raw_value = line.partition(":")
            if key in {"VmRSS", "VmHWM"}:
                values[f"{key.lower()}_kib"] = int(raw_value.split()[0])
    except OSError, ValueError, IndexError:
        return {}
    return values


def model_record(config: SpeakTapConfig) -> dict[str, str | int | None]:
    """Return stable model/runtime fields for benchmark reports."""

    profile = config.model_profile
    return {
        "profile": profile.profile_id,
        "profile_status": profile.status.value,
        "architecture": profile.architecture,
        "model": config.asr_model,
        "quantization": config.asr_quantization,
        "provider": config.execution_provider,
        "threads": config.asr_threads,
        "inter_op_threads": config.asr_inter_op_threads,
        "execution_mode": config.asr_execution_mode,
        "expected_memory_mib": profile.expected_memory_mib,
    }
