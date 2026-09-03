#!/usr/bin/env python3
"""Mutation check for ``test_golden_rule_instrumentation``.

A green suite is only evidence if it can go red.  Each mutation below undoes
exactly one of the guarantees the suite claims to protect; the named test group
must fail.  Every mutation is reverted from a byte-for-byte backup and the
revert is verified by md5 before the next one runs.

Run it with the same interpreter you run the tests with::

    python mutation_check_golden_rule_instrumentation.py

Exit code 0 means: baseline green, every mutation caught, all files restored,
suite green again.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
LEARNING = TESTS.parent
PY = sys.executable
TEST_MODULE = "test_golden_rule_instrumentation"

GRD = LEARNING / "utils" / "golden_rule_diagnostics.py"
EGR = LEARNING / "eval_golden_rule.py"
POL = LEARNING / "policies.py"

# (label, file, exact text to replace, replacement, test group that must fail)
MUTATIONS = [
    (
        "M1 spec2.2  terminated path computes no diagnostics",
        GRD,
        "        if diagnose_on_terminated:",
        "        if False:  # MUTATED",
        "TestTerminatedDiagnostics",
    ),
    (
        "M2 spec2.3  not-evaluated folded into fail",
        GRD,
        "    verdicts: Dict[str, str] = {branch: VERDICT_NOT_EVALUATED for branch in BRANCH_ORDER}",
        "    verdicts: Dict[str, str] = {branch: VERDICT_FAIL for branch in BRANCH_ORDER}  # MUTATED",
        "TestBranchVerdicts",
    ),
    (
        "M3 spec2.4  per-demo reload removed",
        EGR,
        "        policy.load_demo(demo_id=demo_id, task_id=task_id)",
        "        pass  # MUTATED: reload removed",
        "TestPerDemoReload",
    ),
    (
        "M4 spec2.1  snapshot reports source defaults, not resolved values",
        EGR,
        "                str(k): float(v) for k, v in self.get_primitive_success_thresholds().items()",
        '                str(k): float(v) for k, v in {"base_pos": 0.15, "yaw": 0.35, '
        '"eef_pos": 0.12, "gripper_qpos": 0.03, "std_joint_qpos_rmse": 0.25, '
        '"joint_qpos_rmse": 0.25}.items()  # MUTATED',
        "TestThresholdDump",
    ),
    (
        "M5 spec2.4  parquet path observable removed",
        POL,
        "        self.loaded_parquet_path = str(parquet_path)",
        "        pass  # MUTATED",
        "TestPerDemoReload",
    ),
    (
        "M6 verdict preservation  criterion drifts to a strict comparison",
        GRD,
        "        elif std_rmse <= std_thr:",
        "        elif std_rmse < std_thr:  # MUTATED",
        "TestVerdictPreservation",
    ),
    (
        "M7 atomic load  identity committed before the parquet is validated",
        POL,
        "        demo_id = str(demo_id).zfill(8)\n        task_id = int(task_id)\n        parquet_path =",
        "        demo_id = str(demo_id).zfill(8)\n        task_id = int(task_id)\n"
        "        self.demo_id = demo_id  # MUTATED\n        self.task_id = task_id  # MUTATED\n"
        "        parquet_path =",
        "TestPerDemoReload",
    ),
    (
        "M8 setup guard  failed rebind no longer refuses the episode",
        EGR,
        '            self._setup_error = f"replay_reload_failed: {type(exc).__name__}: {exc}"\n'
        "            return False",
        "            self._setup_error = None  # MUTATED\n            pass  # MUTATED",
        "TestSetupEpisodeGuard",
    ),
]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run(target: str) -> int:
    return subprocess.run(
        [PY, "-m", "unittest", target],
        cwd=str(TESTS),
        capture_output=True,
        text=True,
    ).returncode


def main() -> int:
    backup_dir = Path(tempfile.mkdtemp(prefix="gtr_mutation_backup_"))
    originals = {p: backup_dir / p.name for p in (GRD, EGR, POL)}
    for src, dst in originals.items():
        shutil.copy2(src, dst)
    baseline_md5 = {p: md5(p) for p in originals}

    print("=== baseline: unmutated suite must be green ===")
    baseline_rc = run(TEST_MODULE)
    print(f"baseline rc={baseline_rc} ({'GREEN' if baseline_rc == 0 else 'RED'})")
    if baseline_rc != 0:
        print("ABORT: baseline is not green; mutation results would be meaningless")
        return 2

    print("\n=== mutations: each must turn its group RED ===")
    results = []
    for label, path, old, new, group in MUTATIONS:
        text = path.read_text()
        hits = text.count(old)
        if hits != 1:
            print(f"  [{label}] pattern hits={hits} (want 1) -> SKIP")
            results.append((label, group, f"SKIP hits={hits}", False))
            continue
        path.write_text(text.replace(old, new, 1))
        rc = run(f"{TEST_MODULE}.{group}")
        shutil.copy2(originals[path], path)
        restored = md5(path) == baseline_md5[path]
        caught = rc != 0
        results.append((label, group, f"rc={rc}", caught))
        print(
            f"  [{label}]\n"
            f"      group={group} rc={rc} -> "
            f"{'CAUGHT (red)' if caught else 'MISSED (still green)'}"
            f"   restore_verified={restored}"
        )
        if not restored:
            print("  ABORT: failed to restore original file")
            return 3

    print("\n=== post-check: files byte-identical to backup ===")
    for p in originals:
        print(f"  {p.name}: restored={md5(p) == baseline_md5[p]}")

    print("\n=== final: suite green again after all reverts ===")
    final_rc = run(TEST_MODULE)
    print(f"final rc={final_rc} ({'GREEN' if final_rc == 0 else 'RED'})")

    missed = [r for r in results if not r[3]]
    print(f"\nSUMMARY: {len(results) - len(missed)}/{len(results)} mutations caught")
    for label, group, detail, caught in results:
        print(f"  {'OK  ' if caught else 'MISS'}  {label}  [{group}] {detail}")
    shutil.rmtree(backup_dir, ignore_errors=True)
    return 0 if (not missed and final_rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
