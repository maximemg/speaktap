from __future__ import annotations

from speaktap.activity.base import SpeechActivity
from speaktap.domain import AudioChunk, AudioFrame, CutReason
from speaktap.segmentation.adaptive import AdaptiveChunkPolicy


def _frame(index: int) -> AudioFrame:
    return AudioFrame(
        pcm_s16le=b"\x01\x00" * 320,
        sample_rate=16_000,
        channels=1,
        start_ms=index * 20,
    )


def _activity(speech: bool) -> SpeechActivity:
    return SpeechActivity(is_speech=speech, probability=0.9 if speech else 0.1)


def _policy(
    *,
    target_seconds: float = 0.20,
    max_seconds: float = 0.30,
) -> AdaptiveChunkPolicy:
    return AdaptiveChunkPolicy(
        min_chunk_seconds=0.10,
        target_chunk_seconds=target_seconds,
        max_chunk_seconds=max_seconds,
        silence_milliseconds=40,
        padding_milliseconds=20,
        forced_cut_overlap_milliseconds=40,
    )


def test_discards_leading_silence_and_cuts_after_speech_pause() -> None:
    policy = _policy(target_seconds=0.10)
    policy.reset("session")
    chunks: list[AudioChunk] = []
    pattern = [False, False, True, True, True, True, True, False, False]
    for index, speech in enumerate(pattern):
        chunks.extend(policy.push(_frame(index), _activity(speech)))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.cut_reason is CutReason.SILENCE
    assert chunk.start_ms == 20
    assert chunk.end_ms == 160
    assert chunk.speech_ms == 100


def test_forced_cut_retains_prefix_overlap_for_next_chunk() -> None:
    policy = _policy(max_seconds=0.20)
    policy.reset("session")
    chunks: list[AudioChunk] = []
    for index in range(18):
        chunks.extend(policy.push(_frame(index), _activity(True)))
    chunks.extend(policy.finish())

    assert len(chunks) == 2
    assert chunks[0].cut_reason is CutReason.FORCED_MAX
    assert chunks[0].overlap_ms == 0
    assert chunks[1].overlap_ms == 40
    assert chunks[1].start_ms == chunks[0].end_ms - 40


def test_normal_pause_cuts_after_minimum_duration() -> None:
    policy = _policy(target_seconds=0.20)
    policy.reset("session")
    chunks: list[AudioChunk] = []
    pattern = [True, True, True, True, True, False, False]
    for index, speech in enumerate(pattern):
        chunks.extend(policy.push(_frame(index), _activity(speech)))

    assert len(chunks) == 1
    assert chunks[0].cut_reason is CutReason.SILENCE
    assert policy.finish() == ()


def test_target_duration_accepts_a_shorter_pause() -> None:
    policy = AdaptiveChunkPolicy(
        min_chunk_seconds=0.10,
        target_chunk_seconds=0.20,
        max_chunk_seconds=0.40,
        silence_milliseconds=80,
        padding_milliseconds=20,
        forced_cut_overlap_milliseconds=40,
    )
    policy.reset("session")
    chunks: list[AudioChunk] = []
    pattern = [True] * 10 + [False, False]
    for index, speech in enumerate(pattern):
        chunks.extend(policy.push(_frame(index), _activity(speech)))

    assert len(chunks) == 1
    assert chunks[0].cut_reason is CutReason.SILENCE


def test_finish_ignores_silence_only_session() -> None:
    policy = _policy()
    policy.reset("session")
    for index in range(20):
        policy.push(_frame(index), _activity(False))

    assert policy.finish() == ()
