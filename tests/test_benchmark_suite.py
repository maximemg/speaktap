from __future__ import annotations

from pathlib import Path

from speaktap.benchmark.suite import (
    ProfileRun,
    benchmark_command,
    write_suite_report,
)


def test_benchmark_command_selects_profile_without_installing_it(tmp_path: Path) -> None:
    command = benchmark_command(
        profile_id="canary-1b-v2-int8",
        datasets=[tmp_path / "recordings"],
        limit=5,
        passes=1,
        long_seconds="",
        results_dir=tmp_path / "results",
    )

    assert "--profile" in command
    assert command[command.index("--profile") + 1] == "canary-1b-v2-int8"
    assert "speaktap.benchmark.benchmark" in command


def test_suite_report_compares_success_and_failure(tmp_path: Path) -> None:
    report = tmp_path / "REPORT.md"
    summaries = {
        "parakeet-tdt-v3-int8": {
            "totals": {
                "wer_percent": 5.5,
                "latency_median_ms": 1000,
                "latency_p90_ms": 1500,
            },
            "datasets": {
                "recordings": {
                    "clips": 2,
                    "wer_percent": 5.5,
                }
            },
            "long_sessions": [],
            "memory": {"vmrss_kib": 1024},
        }
    }
    runs = [
        ProfileRun(
            profile_id="parakeet-tdt-v3-int8",
            status="ok",
            return_code=0,
            results_dir="one",
        ),
        ProfileRun(
            profile_id="canary-1b-v2-int8",
            status="failed",
            return_code=1,
            results_dir="two",
            error="model failed to load",
        ),
    ]

    write_suite_report(report, summaries=summaries, runs=runs)

    text = report.read_text()
    assert "| parakeet-tdt-v3-int8 | ok | 5.50%" in text
    assert "`canary-1b-v2-int8`: model failed to load" in text
