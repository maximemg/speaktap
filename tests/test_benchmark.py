from __future__ import annotations

from pathlib import Path

from speaktap.benchmark.benchmark import (
    LongSessionResult,
    audio_language,
    build_long_session,
    normalize,
    percentile,
    score,
    select_warmup_file,
    validate_dataset,
    write_report,
)


def test_normalize_and_word_error_rate() -> None:
    result = score("Hello, brave new world!", "hello new world")

    assert normalize("Hello, WORLD!") == ["hello", "world"]
    assert result.errors == 1
    assert result.reference_words == 4
    assert result.wer_percent == 25


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([10, 20, 30, 40, 50], 0.90) == 50
    assert percentile([], 0.90) == 0


def test_long_session_preserves_complete_utterance_references(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one.raw"
    second = tmp_path / "two.raw"
    first.write_bytes(b"\x01\x00" * 1_600)
    second.write_bytes(b"\x02\x00" * 1_600)

    payload, reference, count = build_long_session(
        [first, second],
        {"one": "first sentence", "two": "second sentence"},
        target_seconds=1,
        pause_milliseconds=100,
    )

    assert count == 5
    assert len(payload) / (16_000 * 2) == 1
    assert reference == (
        "first sentence second sentence first sentence second sentence first sentence"
    )


def test_report_supports_long_session_only_run(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    result = LongSessionResult(
        target_seconds=60,
        audio_ms=60_000,
        source_utterances=10,
        post_stop_ms=1_000,
        wer_percent=3.0,
        chunks=9,
        silence_cuts=8,
        forced_cuts=1,
        overlap_chunks=1,
        queue_wait_max_ms=0,
        asr_total_ms=12_000,
    )

    write_report(
        report,
        summary={
            "datasets": {},
            "profile": {"profile_id": "parakeet-tdt-v3-int8"},
            "totals": {
                "silence_cuts": 8,
                "forced_cuts": 1,
                "overlap_chunks": 1,
                "queue_wait_max_ms": 0,
            },
            "memory": {"vmrss_kib": 1024},
        },
        long_sessions=[result],
    )

    text = report.read_text()
    assert "| 60s | 60.0s | 9 | 1000ms | 3.00% | 0ms |" in text
    assert "- Silence cuts: 8" in text
    assert "- Forced cuts: 1" in text
    assert "- Overlap-bearing chunks: 1" in text


def test_validate_user_recording_dataset(tmp_path: Path) -> None:
    (tmp_path / "reference.txt").write_text("sample-one exact dictated text\n")
    (tmp_path / "sample-one.raw").write_bytes(b"\x00\x00" * 320)

    references, audio_files = validate_dataset(tmp_path, limit=10)

    assert references == {"sample-one": "exact dictated text"}
    assert audio_files == [tmp_path / "sample-one.raw"]


def test_audio_language_uses_metadata_or_filename_prefix(tmp_path: Path) -> None:
    audio = tmp_path / "fr-sample.raw"

    assert audio_language(tmp_path, audio) == "fr"

    (tmp_path / "metadata.env").write_text("DATASET_LANGUAGE=en\n")
    assert audio_language(tmp_path, audio) == "en"


def test_warmup_uses_shortest_clip_instead_of_first_name(tmp_path: Path) -> None:
    long_clip = tmp_path / "a-long.raw"
    short_clip = tmp_path / "z-short.raw"
    long_clip.write_bytes(b"\x00\x00" * 32_000)
    short_clip.write_bytes(b"\x00\x00" * 8_000)

    assert select_warmup_file([long_clip, short_clip]) == short_clip
