"""Speech-aware bounded chunking with forced-cut overlap."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..activity.base import SpeechActivity
from ..domain import AudioChunk, AudioFrame, CutReason


@dataclass(frozen=True, slots=True)
class _AnalyzedFrame:
    frame: AudioFrame
    activity: SpeechActivity


class AdaptiveChunkPolicy:
    """Trim leading silence and emit speech chunks at silence or a hard limit."""

    def __init__(
        self,
        *,
        min_chunk_seconds: float,
        target_chunk_seconds: float,
        max_chunk_seconds: float,
        silence_milliseconds: int,
        padding_milliseconds: int,
        forced_cut_overlap_milliseconds: int,
    ) -> None:
        self._min_ms = round(min_chunk_seconds * 1000)
        self._target_ms = round(target_chunk_seconds * 1000)
        self._max_ms = round(max_chunk_seconds * 1000)
        self._silence_ms = silence_milliseconds
        self._padding_ms = padding_milliseconds
        self._forced_overlap_ms = forced_cut_overlap_milliseconds
        self.reset("")

    def reset(self, session_id: str) -> None:
        self._session_id = session_id
        self._sequence = 0
        self._buffer: list[_AnalyzedFrame] = []
        self._pre_roll: deque[_AnalyzedFrame] = deque()
        self._has_speech = False
        self._speech_after_overlap = False
        self._trailing_silence_ms = 0.0
        self._prefix_overlap_ms = 0

    def push(
        self,
        frame: AudioFrame,
        activity: SpeechActivity,
    ) -> tuple[AudioChunk, ...]:
        analyzed = _AnalyzedFrame(frame, activity)
        if not self._buffer:
            if not activity.is_speech:
                self._append_pre_roll(analyzed)
                return ()
            self._buffer = [*self._pre_roll, analyzed]
            self._pre_roll.clear()
            self._has_speech = True
            self._speech_after_overlap = True
            self._recalculate_trailing_silence()
        else:
            self._buffer.append(analyzed)
            self._has_speech = self._has_speech or activity.is_speech
            self._speech_after_overlap = self._speech_after_overlap or activity.is_speech
            self._trailing_silence_ms = (
                0.0 if activity.is_speech else self._trailing_silence_ms + frame.duration_ms
            )

        duration_ms = self._duration_ms(self._buffer)
        if duration_ms >= self._max_ms:
            chunk = self._finalize(CutReason.FORCED_MAX)
            self._retain_forced_overlap()
            return (chunk,)
        required_silence_ms = float(self._silence_ms)
        if duration_ms >= self._target_ms:
            required_silence_ms = max(
                frame.duration_ms,
                self._silence_ms / 2,
            )
        if (
            duration_ms >= self._min_ms
            and self._has_speech
            and self._trailing_silence_ms >= required_silence_ms
        ):
            chunk = self._finalize(CutReason.SILENCE, trim_trailing=True)
            self._clear_active()
            return (chunk,)
        return ()

    def finish(self) -> tuple[AudioChunk, ...]:
        if (
            not self._buffer
            or not self._has_speech
            or (self._prefix_overlap_ms and not self._speech_after_overlap)
        ):
            self._clear_active()
            return ()
        chunk = self._finalize(CutReason.STOP, trim_trailing=True)
        self._clear_active()
        return (chunk,)

    def _append_pre_roll(self, analyzed: _AnalyzedFrame) -> None:
        self._pre_roll.append(analyzed)
        while (
            len(self._pre_roll) > 1 and self._duration_ms(list(self._pre_roll)) > self._padding_ms
        ):
            self._pre_roll.popleft()

    def _finalize(
        self,
        reason: CutReason,
        *,
        trim_trailing: bool = False,
    ) -> AudioChunk:
        selected = self._trim_trailing(self._buffer) if trim_trailing else self._buffer
        first = selected[0].frame
        last = selected[-1].frame
        end_ms = round(last.end_ms)
        start_ms = first.start_ms
        speech_ms = min(
            end_ms - start_ms,
            round(sum(item.frame.duration_ms for item in selected if item.activity.is_speech)),
        )
        overlap_ms = min(self._prefix_overlap_ms, end_ms - start_ms - 1)
        chunk = AudioChunk(
            session_id=self._session_id,
            sequence=self._sequence,
            pcm_s16le=b"".join(item.frame.pcm_s16le for item in selected),
            sample_rate=first.sample_rate,
            channels=first.channels,
            start_ms=start_ms,
            end_ms=end_ms,
            speech_ms=speech_ms,
            cut_reason=reason,
            overlap_ms=max(0, overlap_ms),
        )
        self._sequence += 1
        return chunk

    def _trim_trailing(
        self,
        frames: list[_AnalyzedFrame],
    ) -> list[_AnalyzedFrame]:
        last_speech = max(index for index, item in enumerate(frames) if item.activity.is_speech)
        end = last_speech + 1
        padding = 0.0
        while end < len(frames) and padding < self._padding_ms:
            padding += frames[end].frame.duration_ms
            end += 1
        return frames[:end]

    def _retain_forced_overlap(self) -> None:
        retained: list[_AnalyzedFrame] = []
        duration = 0.0
        for item in reversed(self._buffer):
            retained.append(item)
            duration += item.frame.duration_ms
            if duration >= self._forced_overlap_ms:
                break
        retained.reverse()
        self._buffer = retained
        self._prefix_overlap_ms = round(duration)
        self._has_speech = any(item.activity.is_speech for item in retained)
        self._speech_after_overlap = False
        self._recalculate_trailing_silence()

    def _clear_active(self) -> None:
        self._buffer = []
        self._pre_roll.clear()
        self._has_speech = False
        self._speech_after_overlap = False
        self._trailing_silence_ms = 0.0
        self._prefix_overlap_ms = 0

    def _recalculate_trailing_silence(self) -> None:
        self._trailing_silence_ms = 0.0
        for item in reversed(self._buffer):
            if item.activity.is_speech:
                break
            self._trailing_silence_ms += item.frame.duration_ms

    @staticmethod
    def _duration_ms(frames: list[_AnalyzedFrame]) -> float:
        if not frames:
            return 0.0
        return frames[-1].frame.end_ms - frames[0].frame.start_ms
