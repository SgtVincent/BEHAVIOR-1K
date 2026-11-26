"""Checkpoint <-> stats pairing provenance, verified from the private manifest.

This module only validates recorded paths, hashes, shapes, and metadata
labels. It never opens weights, loads a model, or recomputes dataset
statistics. The old checkpoint must stay paired with its old stats; swapping in
a new run's stats is refused structurally.

The 475-episode (stats_full) vs 800-member (training intake) difference is a
provenance reconciliation item: it is surfaced as ``provenance_pending`` and
must **not** be treated as proof that the stats are wrong, nor justify a
stats swap before sign-off.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .errors import Port2026Error

EXPECTED_STATE_ACTION_WIDTH = 45
EXPECTED_251101_ACTION_SHAPE = [50, 45]


@dataclass(frozen=True)
class LineageReport:
    checkpoint_index_sha: Optional[str]
    stats_full_sha: Optional[str]
    stats_251101_sha: Optional[str]
    checks: List[str]
    problems: List[str]
    provenance_status: str

    @property
    def ok(self) -> bool:
        return not self.problems


def _stat_arrays(record: Dict[str, Any], stat_key: str) -> Dict[str, Any]:
    return record.get("shapes", {}).get(stat_key, {})


def _check_width(record, stat_key, array_names, expected, problems, checks):
    arrays = _stat_arrays(record, stat_key)
    for name in array_names:
        actual = arrays.get(name)
        if actual != expected:
            problems.append(f"stats {stat_key}.{name} shape {actual} != {expected}")
        else:
            checks.append(f"stats {stat_key}.{name} shape {expected}")


def verify_stats_pairing(manifest) -> LineageReport:
    """Validate the recorded checkpoint/stats pairing against a loaded manifest."""
    checkpoint = getattr(manifest, "checkpoint", {})
    stats = getattr(manifest, "stats", {})
    training = getattr(manifest, "training_source", {})

    problems: List[str] = []
    checks: List[str] = []

    ckpt_sha = checkpoint.get("index_sha256")
    if not ckpt_sha:
        problems.append("checkpoint index sha256 missing")
    else:
        checks.append("checkpoint index sha256 recorded")
    if checkpoint.get("weight_payload_rehash") is False:
        checks.append("weight payload full rehash still UNKNOWN (flagged, not claimed)")

    full = stats.get("stats_full", {})
    full_sha = full.get("sha256")
    if not full_sha:
        problems.append("stats_full sha256 missing")
    else:
        checks.append("stats_full sha256 recorded")
    metadata = full.get("metadata", {})
    if metadata.get("schema") != "b1k_full_stats_v1":
        problems.append(f"stats_full schema {metadata.get('schema')!r} != b1k_full_stats_v1")
    else:
        checks.append("stats_full schema b1k_full_stats_v1")
    for stat_key in ("observation.state", "action"):
        _check_width(full, stat_key, ["mean", "std", "q01", "q99"],
                     [EXPECTED_STATE_ACTION_WIDTH], problems, checks)
    if not metadata.get("producer_sha256"):
        problems.append("stats_full producer_sha256 missing")
    else:
        checks.append("stats_full producer_sha256 recorded")

    old = stats.get("stats_251101", {})
    old_sha = old.get("sha256")
    if not old_sha:
        problems.append("stats_251101 sha256 missing")
    else:
        checks.append("stats_251101 sha256 recorded")
    _check_width(old, "action", ["q01", "q99"], EXPECTED_251101_ACTION_SHAPE, problems, checks)

    episodes = metadata.get("episodes")
    members = training.get("dataset_members", "")
    provenance = "reconciled"
    if episodes is not None and "800" in str(members) and int(episodes) != 800:
        provenance = "provenance_pending"
        checks.append(f"episodes {episodes} vs 800 members: sample-choice reconciliation pending; stats not swapped")
    if full.get("provenance_flag"):
        checks.append("provenance flag preserved from manifest")

    return LineageReport(
        checkpoint_index_sha=ckpt_sha,
        stats_full_sha=full_sha,
        stats_251101_sha=old_sha,
        checks=checks,
        problems=problems,
        provenance_status=provenance,
    )


def require_paired_stats(manifest) -> LineageReport:
    report = verify_stats_pairing(manifest)
    if not report.ok:
        raise Port2026Error("checkpoint/stats pairing invalid: " + "; ".join(report.problems))
    return report
