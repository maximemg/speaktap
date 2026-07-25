"""Benchmark one SpeakTap ASR profile on reproducible PCM datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..capture import AudioFileSource
from ..config import SpeakTapConfig
from ..diagnostics import model_record
from ..domain import AudioFrame, TranscriptionResult
from ..factory import make_pipeline
from ..profiles import get_asr_profile
from ..transcription import OnnxAsrBackend

_NORMALIZE = re.compile(r"[^\w']+", flags=re.UNICODE)
_FRAME_MS = 20
_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class Score:
    errors: int
    reference_words: int

    @property
    def wer_percent(self) -> float:
        return self.errors * 100.0 / self.reference_words if self.reference_words else 0.0


@dataclass(frozen=True, slots=True)
class ClipResult:
    dataset: str
    utterance_id: str
    language: str
    audio_ms: int
    reference: str
    transcript: str
    elapsed_ms: int
    post_stop_ms: int
    errors: int
    reference_words: int
    chunks: int
    silence_cuts: int
    forced_cuts: int
    overlap_chunks: int
    queue_wait_max_ms: int
    asr_total_ms: int


@dataclass(frozen=True, slots=True)
class LongSessionResult:
    target_seconds: int
    audio_ms: int
    source_utterances: int
    post_stop_ms: int
    wer_percent: float
    chunks: int
    silence_cuts: int
    forced_cuts: int
    overlap_chunks: int
    queue_wait_max_ms: int
    asr_total_ms: int


class TimedPcmSource:
    """Yield PCM frames at wall-clock speed to emulate microphone capture."""

    def __init__(self, payload: bytes, *, realtime: bool) -> None:
        self._payload = payload
        self._realtime = realtime
        self._stopping = threading.Event()
        self._started_at = 0.0

    def start(self) -> None:
        self._stopping.clear()
        self._started_at = time.monotonic()

    def frames(self) -> Iterator[AudioFrame]:
        frame_bytes = _SAMPLE_RATE * _FRAME_MS // 1000 * 2
        for sequence, offset in enumerate(
            range(0, len(self._payload) - frame_bytes + 1, frame_bytes)
        ):
            if self._stopping.is_set():
                break
            if self._realtime:
                deadline = self._started_at + (sequence + 1) * _FRAME_MS / 1000
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._stopping.wait(remaining)
                if self._stopping.is_set():
                    break
            yield AudioFrame(
                pcm_s16le=self._payload[offset : offset + frame_bytes],
                sample_rate=_SAMPLE_RATE,
                channels=1,
                start_ms=sequence * _FRAME_MS,
            )

    def stop(self) -> None:
        self._stopping.set()


def normalize(text: str) -> list[str]:
    return _NORMALIZE.sub(" ", text.casefold()).strip().split()


def score(reference: str, hypothesis: str) -> Score:
    expected = normalize(reference)
    actual = normalize(hypothesis)
    previous = list(range(len(actual) + 1))
    for expected_index, expected_word in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_word in enumerate(actual, start=1):
            substitution = previous[actual_index - 1] + (expected_word != actual_word)
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    substitution,
                )
            )
        previous = current
    return Score(errors=previous[-1], reference_words=len(expected))


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def load_references(dataset: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    for line in (dataset / "reference.txt").read_text().splitlines():
        utterance_id, separator, text = line.partition(" ")
        if separator:
            references[utterance_id] = text.strip()
    return references


def validate_dataset(dataset: Path, *, limit: int) -> tuple[dict[str, str], list[Path]]:
    reference_path = dataset / "reference.txt"
    if not dataset.is_dir():
        raise ValueError(f"dataset directory does not exist: {dataset}")
    if not reference_path.is_file():
        raise ValueError(f"dataset reference file is missing: {reference_path}")
    references = load_references(dataset)
    audio_files = sorted(dataset.glob("*.raw"))[:limit]
    if not audio_files:
        raise ValueError(f"dataset has no 16 kHz mono PCM16 .raw files: {dataset}")
    missing = [path.stem for path in audio_files if path.stem not in references]
    if missing:
        raise ValueError(f"dataset references are missing for: {', '.join(missing)}")
    return references, audio_files


def dataset_language(dataset: Path) -> str:
    metadata = dataset / "metadata.env"
    if not metadata.exists():
        return ""
    for line in metadata.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "DATASET_LANGUAGE":
            return value.strip()
    return ""


def audio_language(dataset: Path, audio_file: Path) -> str:
    configured = dataset_language(dataset)
    if configured:
        return configured
    prefix, separator, _ = audio_file.stem.partition("-")
    if separator and 2 <= len(prefix) <= 3 and prefix.isalpha():
        return prefix.casefold()
    return ""


def audio_duration_ms(path: Path) -> int:
    return round(path.stat().st_size / (_SAMPLE_RATE * 2) * 1000)


def select_warmup_file(audio_files: list[Path]) -> Path:
    if not audio_files:
        raise ValueError("cannot select a warm-up file from an empty dataset")
    return min(audio_files, key=lambda path: (path.stat().st_size, path.name))


def result_metrics(result: TranscriptionResult) -> dict[str, int]:
    return {
        "chunks": len(result.chunks),
        "silence_cuts": sum(chunk.cut_reason.value == "silence" for chunk in result.chunks),
        "forced_cuts": sum(chunk.cut_reason.value == "forced_max" for chunk in result.chunks),
        "overlap_chunks": sum(chunk.overlap_ms > 0 for chunk in result.chunks),
        "queue_wait_max_ms": max((chunk.queue_wait_ms for chunk in result.chunks), default=0),
        "asr_total_ms": sum(chunk.asr_duration_ms for chunk in result.chunks),
    }


def run_file(
    path: Path,
    *,
    config: SpeakTapConfig,
    backend: OnnxAsrBackend,
    language: str,
) -> tuple[TranscriptionResult, int]:
    pipeline = make_pipeline(
        replace(config, language=language or config.language),
        backend,
    )
    started = time.monotonic()
    pipeline.start(
        AudioFileSource(
            path,
            sample_rate=config.sample_rate,
            frame_milliseconds=config.frame_milliseconds,
        ),
        session_id=uuid4().hex,
        cleanup_enabled=False,
    )
    result = pipeline.wait()
    return result, round((time.monotonic() - started) * 1000)


def run_pcm(
    payload: bytes,
    *,
    config: SpeakTapConfig,
    backend: OnnxAsrBackend,
    realtime: bool,
    language: str = "",
) -> TranscriptionResult:
    pipeline = make_pipeline(
        replace(config, language=language or config.language),
        backend,
    )
    pipeline.start(
        TimedPcmSource(payload, realtime=realtime),
        session_id=uuid4().hex,
        cleanup_enabled=False,
    )
    return pipeline.wait()


def summarize_clips(rows: list[ClipResult]) -> dict[str, Any]:
    errors = sum(row.errors for row in rows)
    words = sum(row.reference_words for row in rows)
    elapsed = [row.elapsed_ms for row in rows]
    return {
        "clips": len(rows),
        "reference_words": words,
        "audio_seconds": sum(row.audio_ms for row in rows) / 1000,
        "wer_percent": errors * 100.0 / words if words else 0.0,
        "latency_median_ms": round(statistics.median(elapsed)) if elapsed else 0,
        "latency_p90_ms": percentile(elapsed, 0.90),
        "post_stop_median_ms": (
            round(statistics.median(row.post_stop_ms for row in rows)) if rows else 0
        ),
        "chunks": sum(row.chunks for row in rows),
        "silence_cuts": sum(row.silence_cuts for row in rows),
        "forced_cuts": sum(row.forced_cuts for row in rows),
        "overlap_chunks": sum(row.overlap_chunks for row in rows),
        "queue_wait_max_ms": max((row.queue_wait_max_ms for row in rows), default=0),
    }


def build_long_session(
    audio_files: list[Path],
    references: dict[str, str],
    *,
    target_seconds: int,
    pause_milliseconds: int,
) -> tuple[bytes, str, int]:
    payload = bytearray()
    text: list[str] = []
    silence = b"\x00\x00" * (_SAMPLE_RATE * pause_milliseconds // 1000)
    target_bytes = target_seconds * _SAMPLE_RATE * 2
    tolerance_bytes = 2 * _SAMPLE_RATE * 2
    ordered = sorted(audio_files, key=lambda path: path.stat().st_size)
    source_count = 0
    while len(payload) < target_bytes:
        remaining = target_bytes - len(payload)
        fitting = [
            path
            for path in ordered
            if path.stat().st_size + len(silence) <= remaining + tolerance_bytes
        ]
        if not fitting:
            break
        path = fitting[source_count % len(fitting)]
        payload.extend(path.read_bytes())
        payload.extend(silence)
        text.append(references[path.stem])
        source_count += 1
    return bytes(payload), " ".join(text), source_count


def run_long_session(
    *,
    target_seconds: int,
    audio_files: list[Path],
    references: dict[str, str],
    config: SpeakTapConfig,
    backend: OnnxAsrBackend,
    results_dir: Path,
    language: str,
) -> LongSessionResult:
    payload, reference, source_count = build_long_session(
        audio_files,
        references,
        target_seconds=target_seconds,
        pause_milliseconds=max(800, config.silence_milliseconds + 100),
    )
    prefix = results_dir / f"dictation-{target_seconds}s"
    prefix.with_suffix(".raw").write_bytes(payload)
    Path(f"{prefix}.reference.txt").write_text(reference + "\n")
    result = run_pcm(
        payload,
        config=config,
        backend=backend,
        realtime=True,
        language=language,
    )
    Path(f"{prefix}.transcript.txt").write_text(result.raw_text + "\n")
    result_score = score(reference, result.raw_text)
    metrics = result_metrics(result)
    return LongSessionResult(
        target_seconds=target_seconds,
        audio_ms=round(len(payload) / (_SAMPLE_RATE * 2) * 1000),
        source_utterances=source_count,
        post_stop_ms=result.timings_ms["post_stop"],
        wer_percent=result_score.wer_percent,
        chunks=metrics["chunks"],
        silence_cuts=metrics["silence_cuts"],
        forced_cuts=metrics["forced_cuts"],
        overlap_chunks=metrics["overlap_chunks"],
        queue_wait_max_ms=metrics["queue_wait_max_ms"],
        asr_total_ms=metrics["asr_total_ms"],
    )


def read_memory(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"VmRSS", "VmHWM"}:
                values[f"{key.lower()}_kib"] = int(value.split()[0])
    except OSError, ValueError, IndexError:
        return {}
    return values


def write_clip_results(path: Path, rows: list[ClipResult]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    long_sessions: list[LongSessionResult],
) -> None:
    lines = [
        "# SpeakTap raw ASR benchmark",
        "",
        f"Profile: `{summary['profile']['profile_id']}`",
        "",
        "| Dataset | Clips | Audio | Raw WER | Median throughput | P90 throughput |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, metrics in summary["datasets"].items():
        lines.append(
            f"| {dataset} | {metrics['clips']} | {metrics['audio_seconds']:.1f}s "
            f"| {metrics['wer_percent']:.2f}% "
            f"| {metrics['latency_median_ms']}ms | {metrics['latency_p90_ms']}ms |"
        )
    lines.extend(
        [
            "",
            "## Real-time dictation",
            "",
            "| Target | Audio | Chunks | Post-stop | Raw WER | Queue wait |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in long_sessions:
        lines.append(
            f"| {result.target_seconds}s | {result.audio_ms / 1000:.1f}s "
            f"| {result.chunks} | {result.post_stop_ms}ms "
            f"| {result.wer_percent:.2f}% | {result.queue_wait_max_ms}ms |"
        )
    memory = summary["memory"].get("vmrss_kib", 0) / 1024
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Warm process RSS: {memory:.0f} MiB",
            f"- Silence cuts: {summary['totals']['silence_cuts']}",
            f"- Forced cuts: {summary['totals']['forced_cuts']}",
            f"- Overlap-bearing chunks: {summary['totals']['overlap_chunks']}",
            f"- Maximum measured queue wait: {summary['totals']['queue_wait_max_ms']} ms",
            "",
            "Short-clip latency feeds file frames as quickly as possible and measures "
            "throughput. Real-time sessions measure user-visible post-stop latency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _parse_long_seconds(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--long-seconds must be comma-separated integers") from error
    if any(value <= 0 for value in values):
        raise ValueError("--long-seconds values must be positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(prog="speaktap-benchmark")
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--profile", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--long-seconds", default="30,60")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark/results") / datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    try:
        long_seconds = _parse_long_seconds(args.long_seconds)
        datasets = [(*validate_dataset(path, limit=args.limit), path) for path in args.datasets]
        config = SpeakTapConfig.load()
        if args.profile:
            config = config.with_profile(get_asr_profile(args.profile).profile_id)
    except ValueError as error:
        parser.error(str(error))

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    backend = OnnxAsrBackend(
        profile=config.model_profile,
        provider=config.execution_provider,
        threads=config.asr_threads,
        inter_op_threads=config.asr_inter_op_threads,
        execution_mode=config.asr_execution_mode,
    )
    print(f"Loading {config.model_profile.display_name}...", flush=True)
    backend.load()

    all_rows: list[ClipResult] = []
    dataset_summaries: dict[str, dict[str, Any]] = {}
    for references, audio_files, dataset in datasets:
        warmup = select_warmup_file(audio_files)
        run_file(
            warmup,
            config=config,
            backend=backend,
            language=audio_language(dataset, warmup),
        )
        rows: list[ClipResult] = []
        for _ in range(args.passes):
            for audio_file in audio_files:
                language = audio_language(dataset, audio_file)
                result, elapsed_ms = run_file(
                    audio_file,
                    config=config,
                    backend=backend,
                    language=language,
                )
                result_score = score(references[audio_file.stem], result.raw_text)
                metrics = result_metrics(result)
                rows.append(
                    ClipResult(
                        dataset=dataset.name,
                        utterance_id=audio_file.stem,
                        language=language,
                        audio_ms=audio_duration_ms(audio_file),
                        reference=references[audio_file.stem],
                        transcript=result.raw_text,
                        elapsed_ms=elapsed_ms,
                        post_stop_ms=result.timings_ms["post_stop"],
                        errors=result_score.errors,
                        reference_words=result_score.reference_words,
                        **metrics,
                    )
                )
        write_clip_results(results_dir / f"{dataset.name}.csv", rows)
        dataset_summaries[dataset.name] = summarize_clips(rows)
        all_rows.extend(rows)

    long_sessions: list[LongSessionResult] = []
    first_references, first_audio, first_dataset = datasets[0]
    for target_seconds in long_seconds:
        long_sessions.append(
            run_long_session(
                target_seconds=target_seconds,
                audio_files=first_audio,
                references=first_references,
                config=config,
                backend=backend,
                results_dir=results_dir,
                language=dataset_language(first_dataset),
            )
        )

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "profile": config.model_profile.to_record(),
        "model": model_record(config),
        "datasets": dataset_summaries,
        "totals": summarize_clips(all_rows),
        "long_sessions": [asdict(result) for result in long_sessions],
        "memory": read_memory(os.getpid()),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(results_dir / "REPORT.md", summary=summary, long_sessions=long_sessions)
    backend.close()
    print(f"Report: {results_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
