"""Summarize per-task average success rate from challenge submission JSON files.

Typical usage:

1. Average the public / hidden submissions that match a reference submission:
   python docs/summarize_task_sr.py \
       --submission-file docs/challenge_submissions/standard.public.ACT.Xiamen_University.20251116.json \
       --submission-dir docs/challenge_submissions

2. Average across every submission JSON in a directory:
   python docs/summarize_task_sr.py \
       --submission-dir docs/challenge_submissions \
       --all-submissions
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TaskScores = Dict[str, float]
IdentityKey = Tuple[str, str, str]


@dataclass(frozen=True)
class SubmissionRecord:
    """In-memory view of one submission JSON."""

    path: Path
    team: str
    affiliation: str
    track: str
    testset: str
    date: str
    task_sr: TaskScores

    @property
    def identity_key(self) -> IdentityKey:
        return (self.track, self.team, self.affiliation)


def load_submission(path: Path) -> SubmissionRecord:
    """Load one submission file and extract per-task success rate."""

    with path.open() as f:
        data = json.load(f)

    try:
        task_sr = data["per_task_scores"]["task_sr"]
    except KeyError as exc:
        raise ValueError(f"{path} 缺少 per_task_scores.task_sr 字段") from exc

    return SubmissionRecord(
        path=path.resolve(),
        team=data.get("team", "Unknown"),
        affiliation=data.get("affiliation", "Unknown"),
        track=data.get("track", "unknown"),
        testset=data.get("testset", "unknown"),
        date=data.get("date", ""),
        task_sr=task_sr,
    )


def load_submissions(paths: Iterable[Path]) -> List[SubmissionRecord]:
    """Load a sequence of submission files."""

    return [load_submission(path) for path in sorted(paths)]


def find_matching_submissions(
    reference_file: Path,
    submission_dir: Path,
) -> List[SubmissionRecord]:
    """Find submissions in the directory that match the same team / track."""

    reference = load_submission(reference_file)
    matched_paths = []
    for candidate in sorted(submission_dir.glob("*.json")):
        try:
            submission = load_submission(candidate)
        except ValueError:
            continue
        if submission.identity_key == reference.identity_key:
            matched_paths.append(candidate)

    # Fall back to the reference file itself when the directory does not contain siblings.
    if not matched_paths:
        matched_paths = [reference_file]

    return load_submissions(matched_paths)


def average(values: Sequence[float]) -> float:
    """Return the arithmetic mean for a non-empty sequence."""

    return sum(values) / len(values)


def summarize_task_scores(submissions: Sequence[SubmissionRecord]) -> List[dict]:
    """Build per-task statistics across the selected submissions."""

    task_names = sorted(
        {
            task_name
            for submission in submissions
            for task_name in submission.task_sr.keys()
        }
    )

    rows = []
    for task_name in task_names:
        selected_values = [
            submission.task_sr[task_name]
            for submission in submissions
            if task_name in submission.task_sr
        ]
        public_values = [
            submission.task_sr[task_name]
            for submission in submissions
            if submission.testset == "public" and task_name in submission.task_sr
        ]
        hidden_values = [
            submission.task_sr[task_name]
            for submission in submissions
            if submission.testset == "hidden" and task_name in submission.task_sr
        ]

        rows.append(
            {
                "task": task_name,
                "avg_task_sr": average(selected_values),
                "num_submissions": len(selected_values),
                "public_avg_task_sr": average(public_values) if public_values else None,
                "hidden_avg_task_sr": average(hidden_values) if hidden_values else None,
            }
        )

    rows.sort(key=lambda row: (-row["avg_task_sr"], row["task"]))
    return rows


def format_optional(value: float | None) -> str:
    """Format an optional floating-point number for terminal output."""

    return "" if value is None else f"{value:.4f}"


def print_summary(submissions: Sequence[SubmissionRecord], rows: Sequence[dict]) -> None:
    """Print a human-readable table."""

    print("Selected submissions:")
    for submission in submissions:
        print(
            f"- {submission.path} "
            f"(track={submission.track}, testset={submission.testset}, "
            f"team={submission.team}, affiliation={submission.affiliation}, date={submission.date})"
        )

    print()
    print(
        f"{'task':<45} {'avg_task_sr':>12} {'public_avg':>12} "
        f"{'hidden_avg':>12} {'count':>7}"
    )
    print("-" * 92)
    for row in rows:
        print(
            f"{row['task']:<45} "
            f"{row['avg_task_sr']:>12.4f} "
            f"{format_optional(row['public_avg_task_sr']):>12} "
            f"{format_optional(row['hidden_avg_task_sr']):>12} "
            f"{row['num_submissions']:>7}"
        )


def write_csv(rows: Sequence[dict], output_csv: Path) -> None:
    """Write the summary rows to a CSV file."""

    header = [
        "task",
        "avg_task_sr",
        "public_avg_task_sr",
        "hidden_avg_task_sr",
        "num_submissions",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["task"],
                    f"{row['avg_task_sr']:.6f}",
                    "" if row["public_avg_task_sr"] is None else f"{row['public_avg_task_sr']:.6f}",
                    "" if row["hidden_avg_task_sr"] is None else f"{row['hidden_avg_task_sr']:.6f}",
                    str(row["num_submissions"]),
                ]
            )
        )
    output_csv.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="统计 submission 的每任务平均成功率")
    parser.add_argument(
        "--submission-file",
        type=Path,
        help="参考 submission JSON；默认会在 submission-dir 中寻找同 team/track/affiliation 的 public/hidden 文件",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=Path("docs/challenge_submissions"),
        help="submission JSON 所在目录，默认是 docs/challenge_submissions",
    )
    parser.add_argument(
        "--all-submissions",
        action="store_true",
        help="忽略 submission-file，直接统计 submission-dir 下全部 JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="可选：将结果另存为 CSV 文件",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    submission_dir = args.submission_dir.resolve()

    if args.all_submissions:
        submission_paths = sorted(submission_dir.glob("*.json"))
        if not submission_paths:
            raise FileNotFoundError(f"{submission_dir} 下没有找到任何 JSON 文件")
        submissions = load_submissions(submission_paths)
    else:
        if args.submission_file is None:
            raise ValueError("未指定 --submission-file；如果要统计整个目录，请加 --all-submissions")
        submissions = find_matching_submissions(
            reference_file=args.submission_file.resolve(),
            submission_dir=submission_dir,
        )

    rows = summarize_task_scores(submissions)
    print_summary(submissions, rows)

    if args.output_csv:
        output_csv = args.output_csv.resolve()
        write_csv(rows, output_csv)
        print()
        print(f"CSV saved to: {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow usage such as `... | head` without printing a noisy traceback.
        sys.exit(0)
