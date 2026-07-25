"""Threaded streaming session that keeps capture independent from inference."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .activity.base import SpeechDetector
from .assembly import assemble_transcript
from .capture.base import AudioSource
from .cleanup.base import Cleaner
from .domain import (
    AudioChunk,
    CleanupStatus,
    TranscriptChunk,
    TranscriptionResult,
)
from .segmentation.base import ChunkPolicy
from .transcription.base import AsrBackend

# Identity-only sentinel: it cannot collide with a valid queued chunk and lets
# the single ASR worker drain all preceding work before terminating.
_STOP = object()


@dataclass(frozen=True, slots=True)
class _QueuedChunk:
    chunk: AudioChunk
    enqueued_at: float


class PipelineSession:
    """Run one capture session while reusing a persistent ASR backend."""

    def __init__(
        self,
        *,
        detector: SpeechDetector,
        chunk_policy: ChunkPolicy,
        backend: AsrBackend,
        cleaner: Cleaner,
        language: str = "",
        cleanup_timeout_seconds: float = 2.0,
        max_pending_chunks: int = 8,
        max_recording_seconds: float = 300.0,
    ) -> None:
        self._detector = detector
        self._chunk_policy = chunk_policy
        self._backend = backend
        self._cleaner = cleaner
        self._language = language
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._max_pending_chunks = max_pending_chunks
        self._max_recording_ms = round(max_recording_seconds * 1000)
        self._lock = threading.RLock()
        self._source: AudioSource | None = None
        self._capture_thread: threading.Thread | None = None
        self._asr_thread: threading.Thread | None = None
        self._queue: queue.Queue[_QueuedChunk | object] | None = None
        self._transcripts: dict[int, TranscriptChunk] = {}
        self._capture_error: BaseException | None = None
        self._session_id = ""
        self._cleanup_enabled = False
        self._started_at = 0.0

    def start(
        self,
        source: AudioSource,
        *,
        session_id: str,
        cleanup_enabled: bool,
    ) -> None:
        with self._lock:
            if self._source is not None:
                raise RuntimeError("a pipeline session is already active")
            self._session_id = session_id
            self._cleanup_enabled = cleanup_enabled
            self._capture_error = None
            self._transcripts = {}
            # Each chunk owns PCM bytes. Bounding the queue caps memory when ASR
            # falls behind and deliberately propagates backpressure to capture.
            self._queue = queue.Queue(maxsize=self._max_pending_chunks)
            self._detector.reset()
            self._chunk_policy.reset(session_id)
            source.start()
            self._source = source
            self._started_at = time.monotonic()
            self._asr_thread = threading.Thread(
                target=self._asr_loop,
                name="speaktap-inference",
                daemon=True,
            )
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="speaktap-capture",
                daemon=True,
            )
            self._asr_thread.start()
            self._capture_thread.start()

    def stop(
        self,
        *,
        on_capture_stopped: Callable[[], None] | None = None,
    ) -> TranscriptionResult:
        stop_started = time.monotonic()
        with self._lock:
            source = self._require_source()
        source.stop()
        self._join_capture()
        if on_capture_stopped is not None:
            on_capture_stopped()
        for chunk in self._chunk_policy.finish():
            self._put_chunk(chunk)
        self._finish_asr()
        result = self._build_result(stop_started=stop_started)
        self._reset_runtime()
        if self._capture_error is not None:
            raise RuntimeError(f"audio capture failed: {self._capture_error}")
        return result

    def wait(self) -> TranscriptionResult:
        """Wait for a finite source and finalize it."""

        self._join_capture()
        return self.stop()

    def cancel(self) -> None:
        with self._lock:
            source = self._source
        if source is None:
            return
        source.stop()
        self._join_capture()
        self._finish_asr()
        self._reset_runtime()

    def _capture_loop(self) -> None:
        try:
            source = self._require_source()
            for frame in source.frames():
                activity = self._detector.analyze(frame)
                for chunk in self._chunk_policy.push(frame, activity):
                    self._put_chunk(chunk)
                if frame.end_ms >= self._max_recording_ms:
                    source.stop()
                    break
        except BaseException as error:
            self._capture_error = error

    def _asr_loop(self) -> None:
        work_queue = self._require_queue()
        while True:
            item = work_queue.get()
            try:
                if item is _STOP:
                    return
                if not isinstance(item, _QueuedChunk):
                    raise TypeError("invalid ASR queue item")
                chunk = item.chunk
                queue_wait_ms = round((time.monotonic() - item.enqueued_at) * 1000)
                started = time.monotonic()
                try:
                    asr_result = self._backend.transcribe(
                        chunk,
                        language=self._language,
                    )
                    raw_text = asr_result.text
                    words = asr_result.words
                    error = None
                except Exception as exception:
                    raw_text = ""
                    words = ()
                    error = str(exception)
                transcript = TranscriptChunk(
                    session_id=chunk.session_id,
                    sequence=chunk.sequence,
                    audio_start_ms=chunk.start_ms,
                    audio_end_ms=chunk.end_ms,
                    cut_reason=chunk.cut_reason,
                    raw_text=raw_text,
                    asr_duration_ms=round((time.monotonic() - started) * 1000),
                    speech_ms=chunk.speech_ms,
                    overlap_ms=chunk.overlap_ms,
                    queue_wait_ms=queue_wait_ms,
                    words=words,
                    error=error,
                )
                with self._lock:
                    self._transcripts[chunk.sequence] = transcript
            finally:
                work_queue.task_done()

    def _build_result(self, *, stop_started: float) -> TranscriptionResult:
        # _finish_asr() has joined the only writer before this method is called,
        # so _transcripts is stable and can be read without taking _lock.
        chunks = tuple(transcript for _, transcript in sorted(self._transcripts.items()))
        assembly_started = time.monotonic()
        raw_text = assemble_transcript(chunks)
        assembly_ms = round((time.monotonic() - assembly_started) * 1000)
        # Skip the stage entirely when disabled. Running it and discarding the
        # result burned CPU on every session and reported a cleanup time of
        # zero, which hid the cost from the diagnostics that would reveal it.
        cleaned_text: str | None = None
        cleanup_status = CleanupStatus.DISABLED
        cleanup_ms = 0
        if self._cleanup_enabled:
            cleanup_started = time.monotonic()
            cleanup = self._cleaner.clean(
                raw_text,
                chunks,
                timeout_seconds=self._cleanup_timeout_seconds,
            )
            cleanup_ms = round((time.monotonic() - cleanup_started) * 1000)
            cleanup_status = cleanup.status
            if cleanup.status is CleanupStatus.SUCCESS:
                cleaned_text = cleanup.text
        finished_at = time.monotonic()
        return TranscriptionResult(
            session_id=self._session_id,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            cleanup_status=cleanup_status,
            chunks=chunks,
            timings_ms={
                "session": round((finished_at - self._started_at) * 1000),
                "post_stop": round((finished_at - stop_started) * 1000),
                "assembly": assembly_ms,
                "cleanup": cleanup_ms,
                "asr_total": sum(chunk.asr_duration_ms for chunk in chunks),
                "asr_max": max(
                    (chunk.asr_duration_ms for chunk in chunks),
                    default=0,
                ),
                "queue_wait_max": max(
                    (chunk.queue_wait_ms for chunk in chunks),
                    default=0,
                ),
            },
        )

    def _put_chunk(self, chunk: AudioChunk) -> None:
        self._require_queue().put(_QueuedChunk(chunk=chunk, enqueued_at=time.monotonic()))

    def _finish_asr(self) -> None:
        work_queue = self._require_queue()
        # FIFO ordering puts the sentinel after every submitted chunk. join()
        # waits for task_done() on both the chunks and sentinel; joining the
        # thread then establishes that no transcript writer remains.
        work_queue.put(_STOP)
        work_queue.join()
        asr_thread = self._asr_thread
        if asr_thread is not None:
            asr_thread.join()

    def _join_capture(self) -> None:
        capture_thread = self._capture_thread
        if capture_thread is not None and capture_thread is not threading.current_thread():
            capture_thread.join()

    def _reset_runtime(self) -> None:
        with self._lock:
            self._source = None
            self._capture_thread = None
            self._asr_thread = None
            self._queue = None

    def _require_source(self) -> AudioSource:
        if self._source is None:
            raise RuntimeError("pipeline session is not active")
        return self._source

    def _require_queue(self) -> queue.Queue[_QueuedChunk | object]:
        if self._queue is None:
            raise RuntimeError("pipeline queue is not active")
        return self._queue
