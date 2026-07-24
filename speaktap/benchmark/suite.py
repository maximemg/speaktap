"""Run registered SpeakTap ASR profiles sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..profiles import benchmark_profile_ids, get_asr_profile


@dataclass(frozen=True, slots=True)
class ProfileRun:
    profile_id: str
    status: str
    return_code: int
    results_dir: str
    error: str | None = None


def benchmark_command(
    *,
    profile_id: str,
    datasets: list[Path],
    limit: int,
    passes: int,
    long_seconds: str,
    results_dir: Path,
) -> list[str]:
    get_asr_profile(profile_id)
    return [
        sys.executable,
        "-m",
        "speaktap.benchmark.benchmark",
        *(str(dataset) for dataset in datasets),
        "--profile",
        profile_id,
        "--limit",
        str(limit),
        "--passes",
        str(passes),
        "--long-seconds",
        long_seconds,
        "--results-dir",
        str(results_dir),
    ]


def write_suite_report(
    path: Path,
    *,
    summaries: dict[str, dict[str, Any]],
    runs: list[ProfileRun],
) -> None:
    lines = [
        "# SpeakTap ASR profile comparison",
        "",
        "Each profile ran in a fresh process. Only one model was loaded at a time, "
        "and the installed profile was not changed.",
        "",
        "| Profile | Run | Raw WER | Median throughput | P90 throughput | Warm RSS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        summary = summaries.get(run.profile_id)
        if summary is None:
            lines.append(f"| {run.profile_id} | {run.status} | — | — | — | — |")
            continue
        totals = summary.get("totals", {})
        memory = summary.get("memory", {}).get("vmrss_kib", 0) / 1024
        lines.append(
            f"| {run.profile_id} | {run.status} "
            f"| {totals.get('wer_percent', 0):.2f}% "
            f"| {totals.get('latency_median_ms', 0)}ms "
            f"| {totals.get('latency_p90_ms', 0)}ms "
            f"| {memory:.0f} MiB |"
        )

    lines.extend(
        [
            "",
            "## Dataset accuracy",
            "",
            "| Profile | Dataset | Clips | Raw WER |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for profile_id, summary in summaries.items():
        for dataset, metrics in summary.get("datasets", {}).items():
            lines.append(
                f"| {profile_id} | {dataset} | {metrics['clips']} "
                f"| {metrics['wer_percent']:.2f}% |"
            )

    lines.extend(
        [
            "",
            "## Real-time dictation",
            "",
            "| Profile | Audio | Chunks | Post-stop | Raw WER | Queue wait |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile_id, summary in summaries.items():
        for result in summary.get("long_sessions", []):
            lines.append(
                f"| {profile_id} | {result['audio_ms'] / 1000:.1f}s "
                f"| {result['chunks']} | {result['post_stop_ms']}ms "
                f"| {result['wer_percent']:.2f}% "
                f"| {result['queue_wait_max_ms']}ms |"
            )

    failures = [run for run in runs if run.status != "ok"]
    if failures:
        lines.extend(["", "## Failed profiles", ""])
        for run in failures:
            lines.append(f"- `{run.profile_id}`: {run.error or 'benchmark failed'}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="speaktap-model-benchmark")
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--profiles",
        default=",".join(benchmark_profile_ids()),
        help="comma-separated registered profile IDs",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--long-seconds", default="30,60")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark/results")
        / f"profiles_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.passes <= 0:
        parser.error("--passes must be positive")
    profile_ids = [value.strip() for value in args.profiles.split(",") if value.strip()]
    if not profile_ids:
        parser.error("--profiles must contain at least one profile")
    try:
        profile_ids = [get_asr_profile(value).profile_id for value in profile_ids]
    except ValueError as error:
        parser.error(str(error))

    root: Path = args.results_dir
    root.mkdir(parents=True, exist_ok=True)
    runs: list[ProfileRun] = []
    summaries: dict[str, dict[str, Any]] = {}
    for profile_id in profile_ids:
        result_dir = root / profile_id
        print(f"\n=== {profile_id} ===", flush=True)
        completed = subprocess.run(
            benchmark_command(
                profile_id=profile_id,
                datasets=args.datasets,
                limit=args.limit,
                passes=args.passes,
                long_seconds=args.long_seconds,
                results_dir=result_dir,
            ),
            check=False,
        )
        summary_path = result_dir / "summary.json"
        if completed.returncode == 0 and summary_path.exists():
            summary: dict[str, Any] = json.loads(summary_path.read_text())
            summaries[profile_id] = summary
            runs.append(ProfileRun(profile_id, "ok", 0, str(result_dir)))
        else:
            runs.append(
                ProfileRun(
                    profile_id=profile_id,
                    status="failed",
                    return_code=completed.returncode,
                    results_dir=str(result_dir),
                    error=f"benchmark exited with code {completed.returncode}",
                )
            )

    suite_summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "profiles": [get_asr_profile(profile_id).to_record() for profile_id in profile_ids],
        "runs": [asdict(run) for run in runs],
        "results": summaries,
    }
    (root / "summary.json").write_text(
        json.dumps(suite_summary, ensure_ascii=False, indent=2) + "\n"
    )
    write_suite_report(root / "REPORT.md", summaries=summaries, runs=runs)
    print(f"\nComparison: {root / 'REPORT.md'}")
    if any(run.status != "ok" for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
