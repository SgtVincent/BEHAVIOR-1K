#!/usr/bin/env python3
"""
Wait for per-task `report.json` files to appear under `eval_logs/` for a list of tasks,
then aggregate them into a combined JSON + Markdown report (like `run_batch_train_eval.sh`).

Usage:
  python scripts/wait_and_aggregate_reports.py --tasks putting_shoes_on_rack,... \
      --out-dir batch_reports_auto_20260128-xxxx --poll 60 --timeout 86400

By default it uses TASKS from the environment (comma-separated) and writes to
`batch_reports_auto_<timestamp>` under repo root.
"""

import argparse
import json
import os
from pathlib import Path

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval_logs"


def find_report_for_task(task: str):
    # Find the latest eval dir for the task and check for report.json
    candidates = sorted(EVAL_ROOT.glob(f"{task}_all_primitives_parallel_*"))
    if not candidates:
        return None, None
    eval_dir = candidates[-1]
    report_path = eval_dir / "report.json"
    if report_path.exists():
        return eval_dir, report_path
    return eval_dir, None


def load_report(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def aggregate_reports(task_reports, out_dir: Path):
    # task_reports: dict task -> (eval_dir Path, report dict)
    combined = {
        "log_dir": str(out_dir),
        "tasks": sorted(list(task_reports.keys())),
        "models": ["act", "dp3", "moma_stage", "wbvima"],
        "results": {},
    }

    for task, (eval_dir, report) in sorted(task_reports.items()):
        if report is None:
            combined["results"][task] = {"status": "MISSING_REPORT", "eval_dir": str(eval_dir) if eval_dir else ""}
            continue
        combined["results"][task] = {
            "status": "SUCCESS",
            "eval_dir": str(eval_dir),
            "models": report.get("models", {}),
            "demo_ids": report.get("demo_ids", []),
            "primitive_max_steps": report.get("primitive_max_steps"),
        }

    out_json = out_dir / "combined_report.json"
    out_md = out_dir / "report.md"
    out_json.write_text(json.dumps(combined, indent=2))

    # Generate Markdown similar to run_batch_train_eval.sh
    lines = []
    lines.append("# BEHAVIOR-1K Batch Train-Eval Report\n")
    lines.append(f"**Log directory:** `{out_dir}`\n")
    lines.append(f"**Tasks ({len(combined['tasks'])}):** {', '.join(f'`{t}`' for t in combined['tasks'])}\n")
    lines.append("**Models:** ACT, DP3, MoMa-STAGE, WBVIMA\n")

    lines.append("## Overall Summary\n")
    lines.append("| Task | Status | ACT | DP3 | MoMa-STAGE | WBVIMA |")
    lines.append("|------|--------|-----|-----|------------|--------|")

    def fmt(s, t):
        if t == 0:
            return "-"
        rate = s / t * 100
        return f"{s}/{t} ({rate:.0f}%)"

    for task in combined["tasks"]:
        r = combined["results"].get(task, {})
        status = r.get("status", "?")
        if "error" in r:
            lines.append(f"| {task} | {status} | - | - | - | - |")
            continue
        row = [task, status]
        for m in combined["models"]:
            mr = r.get("models", {}).get(m, {})
            s = mr.get("total_success", 0)
            t = mr.get("total_trials", 0)
            row.append(fmt(s, t))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Highlights\n")
    for task in combined["tasks"]:
        r = combined["results"].get(task, {})
        if "error" in r or "models" not in r:
            continue
        rates = {}
        for m in combined["models"]:
            mr = r.get("models", {}).get(m, {})
            t = mr.get("total_trials", 0)
            if t > 0:
                rates[m] = mr.get("success_rate", 0)
        if not rates:
            continue
        best_m = max(rates, key=rates.get)
        worst_m = min(rates, key=rates.get)
        lines.append(f"- **{task}**: Best = `{best_m}` ({rates[best_m]*100:.0f}%), Worst = `{worst_m}` ({rates[worst_m]*100:.0f}%)")
    lines.append("")

    lines.append("## Model Comparison (Averaged Across Tasks)\n")
    model_totals = {m: {"s": 0, "t": 0} for m in combined["models"]}
    for task in combined["tasks"]:
        r = combined["results"].get(task, {})
        for m in combined["models"]:
            mr = r.get("models", {}).get(m, {})
            model_totals[m]["s"] += mr.get("total_success", 0)
            model_totals[m]["t"] += mr.get("total_trials", 0)

    lines.append("| Model | Total Success | Total Trials | Overall Success Rate |")
    lines.append("|-------|---------------|--------------|----------------------|")
    for m in combined["models"]:
        s = model_totals[m]["s"]
        t = model_totals[m]["t"]
        rate = (s / t * 100) if t > 0 else 0
        lines.append(f"| {m} | {s} | {t} | {rate:.1f}% |")
    lines.append("")

    # Per-Task Details
    lines.append("## Per-Task Details\n")
    for task in combined["tasks"]:
        r = combined["results"].get(task, {})
        lines.append(f"### {task}\n")
        status = r.get("status", "UNKNOWN")
        lines.append(f"- **Status:** {status}")
        if "error" in r:
            lines.append(f"- **Error:** {r['error']}\n")
            continue
        lines.append(f"- **Eval dir:** `{r.get('eval_dir', 'N/A')}`")
        lines.append(f"- **Demos:** `{', '.join(r.get('demo_ids', []))}`")
        lines.append(f"- **primitive_max_steps:** `{r.get('primitive_max_steps')}`\n")
        lines.append("| Model | Success/Trials | Rate | Result Types |")
        lines.append("|-------|----------------|------|--------------|")
        for m in combined["models"]:
            mr = r.get("models", {}).get(m, {})
            s = mr.get("total_success", 0)
            t = mr.get("total_trials", 0)
            rate = mr.get("success_rate", 0)
            brt = mr.get("by_result_type", {})
            lines.append(f"| {m} | {s}/{t} | {rate:.3f} | {brt} |")
        lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"Wrote: {out_json}\nWrote: {out_md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task list (overrides TASKS env var)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for combined report")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout seconds (0 = no timeout)")
    args = parser.parse_args()

    tasks_env = args.tasks or os.environ.get("TASKS")
    if not tasks_env:
        print("ERROR: need --tasks or TASKS env var (comma-separated)")
        sys.exit(2)
    tasks = [t.strip() for t in tasks_env.split(",") if t.strip()]

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / f"batch_reports_auto_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    poll = args.poll
    timeout = args.timeout

    print(f"Waiting for report.json for tasks: {', '.join(tasks)}")
    start = time.time()
    task_reports = {t: (None, None) for t in tasks}

    while True:
        all_ok = True
        for t in tasks:
            eval_dir, report_path = find_report_for_task(t)
            if report_path and report_path.exists():
                if task_reports[t][1] is None:
                    print(f"Found report for {t}: {report_path}")
                    task_reports[t] = (eval_dir, load_report(report_path))
                # else already loaded
            else:
                all_ok = False
                if eval_dir is None:
                    print(f"No eval directory found yet for {t}")
                else:
                    print(f"Eval dir for {t} exists ({eval_dir}) but report.json missing")

        if all_ok:
            print("All reports found, aggregating...")
            aggregate_reports(task_reports, out_dir)
            print("Aggregation complete")
            break

        if timeout and (time.time() - start) > timeout:
            print("Timeout reached, aggregating available reports (partial)")
            aggregate_reports(task_reports, out_dir)
            break

        time.sleep(poll)


if __name__ == "__main__":
    main()
