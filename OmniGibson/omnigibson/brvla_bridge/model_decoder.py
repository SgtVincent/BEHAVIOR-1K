"""Model physical23 chunk -> fixed-anchor planner targets -> native grid seam.

This module is the decoder body between the frozen server's physical chunk
and the certified planner / native controller. It is **not** a relabeling
boundary: it never turns a physical row into a joint-native23 command. The
joint-native23 tag and consumer guard live in :mod:`action_tag` and are applied
by the caller after this seam assembles a verified native23 vector.

Profile contract (verified against the frozen ``physical23_same_anchor_cumulative``
chunk, not the converted NP episode):

- The chunk rows are already cumulative against the fixed frame0 physical-base
  anchor. The :data:`delta_tf.REPRESENTATION_CUMULATIVE` profile is used; the
  pairwise-delta SE3 accumulator is never applied to these columns.
- Columns follow the frozen transport order: left pose6 ``0:6``, right pose6
  ``6:12``, continuous gripper ``12:14``, body move ``14:17``, chest pose6
  ``17:23``.
- Base velocity and continuous gripper widths come from the model target seam;
  planner defaults are never substituted for them.
- Inactive arms hold the live SDK current qpos, bound by source sha256. An
  unbound inactive group still fails closed; no zero default is emitted.
- A live-scene plan is blocked unless the scope receipt carries the explicit
  live-scene identity, source epoch, and config sha back-link.

Stdlib-only at import. No torch, numpy, omnigibson, simulator, or IK solve.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence

from . import contracts, delta_tf, plan_wrapper, planner_grid, scope
from ._arr import is_finite, to_plain
from .errors import ContractViolation

MODEL_LAYOUT = contracts.SERVER_METADATA_LAYOUT
MAX_CHUNK_ROWS = contracts.CHUNK_SIZE
EXECUTED_HORIZON = contracts.HORIZON

# Source checkpoint identity fields the decoder binds at construction. The
# weight-payload full rehash stays a separate provenance gap; these fields
# index the frozen serving identity, they do not certify weight bytes.
SOURCE_RECEIPT_FIELDS = ("checkpoint_index_sha256", "stats_sha256", "server_sha256")


class ModelDecoderViolation(ContractViolation):
    """A model chunk, anchor, current-q, or source-receipt binding violates the seam."""


def _row_sha256(row: Sequence[float]) -> str:
    blob = json.dumps([float(v) for v in row], sort_keys=True,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _validate_chunk(chunk: Sequence[Sequence[float]]) -> List[List[float]]:
    plain = [list(to_plain(row)) for row in chunk]
    if not plain:
        raise ModelDecoderViolation("model chunk is empty")
    if len(plain) > MAX_CHUNK_ROWS:
        raise ModelDecoderViolation(
            f"chunk rows {len(plain)} > pinned chunk size {MAX_CHUNK_ROWS}")
    rows: List[List[float]] = []
    for row in plain:
        if len(row) != contracts.ACTION23:
            raise ModelDecoderViolation(
                f"chunk row width {len(row)} != {contracts.ACTION23}")
        if not is_finite(row):
            raise ModelDecoderViolation("chunk row contains non-finite values")
        rows.append([float(v) for v in row])
    return rows


def _validate_source_receipt(receipt: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(receipt, dict):
        raise ModelDecoderViolation("source checkpoint receipt is required")
    bound: Dict[str, str] = {}
    for field in SOURCE_RECEIPT_FIELDS:
        value = receipt.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ModelDecoderViolation(f"source receipt missing {field!r}")
        bound[field] = value.strip()
    return bound


class ModelChunkDecoder:
    """Decompose a frozen model chunk into planner targets and native grid.

    Args:
        chunk: model physical23 rows (``N x 23``), same-anchor cumulative.
        anchors: fixed frame0 physical-base pose6 for ``left``/``right``/``chest``.
        dt_seconds: model control step dt (sealed 1/30).
        scope_report: verified :class:`~brvla_bridge.scope.ScopeReport`.
        executor: optional duck-typed planner seam; absent until registered.
        current_q: inactive-arm current-q bindings from
            :func:`native_action.bind_current_q`.
        source_receipt: frozen checkpoint / stats / server identity shas.
        planner_dt: planner trajectory sample period (receipt value).
        physics_dt: evaluator control step (sealed 1/30).
        max_step_delta: closed per-coordinate motion bound for the native grid.
        native_state_receipt: optional live native joint-state snapshot provenance.
    """

    def __init__(
        self,
        *,
        chunk: Sequence[Sequence[float]],
        anchors: Dict[str, Sequence[float]],
        dt_seconds: float,
        scope_report: scope.ScopeReport,
        executor: Optional[Any] = None,
        current_q: Optional[Dict[str, Dict[str, Any]]] = None,
        source_receipt: Optional[Dict[str, Any]] = None,
        planner_dt: float = planner_grid.PLANNER_DT_S,
        physics_dt: float = contracts.CONTROL_DT,
        max_step_delta: Optional[float] = None,
        native_state_receipt: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(scope_report, scope.ScopeReport):
            raise ModelDecoderViolation("verified scope report required")
        scope_report.require()

        rows = _validate_chunk(chunk)
        dt = float(dt_seconds)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ModelDecoderViolation("dt must be finite and positive")

        # Strict cumulative profile: no pairwise accumulation over model rows.
        self._targets = delta_tf.FixedAnchorTargets(
            rows=rows,
            anchors=anchors,
            dt_seconds=dt,
            representation=delta_tf.REPRESENTATION_CUMULATIVE,
        )
        self._scope = scope_report
        self._source = _validate_source_receipt(source_receipt)
        self._row_shas = [_row_sha256(row) for row in rows]
        self._native_state_receipt = dict(native_state_receipt or {})

        self._wrapper = plan_wrapper.ManagedReplayWrapper(
            targets=self._targets,
            scope_report=scope_report,
            executor=executor,
            planner_dt=planner_dt,
            physics_dt=physics_dt,
            current_q=current_q,
            max_step_delta=max_step_delta,
        )

    # -- read-only identity -------------------------------------------------

    @property
    def episode_length(self) -> int:
        return self._targets.length

    def targets_at(self, index: int) -> Dict[str, Any]:
        return self._targets.target(index)

    def bind_current_q(
        self,
        arm: str,
        qpos: Sequence[Any],
        *,
        source_sha256: str,
        source_kind: str = "live_sdk_qpos",
    ) -> None:
        """Bind an inactive arm's live current qpos to hold under collision."""
        self._wrapper.bind_current_q(
            arm, qpos, source_sha256=source_sha256, source_kind=source_kind)

    # -- plan / grid composition --------------------------------------------

    def plan_row(self, index: int) -> Dict[str, Any]:
        return self._wrapper.plan_row(index)

    def assemble_step(self, plan: Dict[str, Any], physics_step: int) -> Dict[str, Any]:
        return self._wrapper.assemble_step(plan, physics_step)

    # -- provenance receipt ---------------------------------------------------

    def receipt(self) -> Dict[str, Any]:
        return {
            "model_layout": MODEL_LAYOUT,
            "representation": delta_tf.REPRESENTATION_CUMULATIVE,
            "rows": self._targets.length,
            "chunk_full_recomputed": self._targets.length == MAX_CHUNK_ROWS,
            "row_sha256": self._row_shas,
            "dt_seconds": self._targets.dt_seconds,
            "anchors": self._targets.anchors(),
            "source": self._source,
            "native_state": self._native_state_receipt,
            "scope": {
                "pass": self._scope.pass_,
                "missing": self._scope.missing,
                "evidence": self._scope.evidence,
            },
            "current_q": {
                group: {"source": entry["source"], "source_kind": entry["source_kind"],
                        "source_sha256": entry["source_sha256"]}
                for group, entry in self._wrapper_current_q().items()
            },
        }

    def _wrapper_current_q(self) -> Dict[str, Dict[str, Any]]:
        return getattr(self._wrapper, "_current_q", {})
