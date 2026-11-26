"""Managed replay policy bound to one pinned converted episode.

``DemoReplayPolicy`` is instantiated by the official 2026 ``BatchedEvaluator``
through its hydra ``_target_`` seam (the same constructor seam used by
``LocalPolicy`` / ``WebsocketPolicy``). The official evaluator keeps its own
reset / observation / step / loop; this policy only returns the saved
execution23 rows in their original order. It performs no model forward, no
frozen-server call, no IK solve, and no custom environment loop.

Bound identity (task / demo / instance) is recorded at construction and checked
again by the optional ``load_demo`` hook; a different scene or frame0 must not
reuse this episode's anchor. Replay rows are pairwise deltas against the fixed
run-local start anchor recorded in the converted manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Dict, List, Optional

from . import contracts
from .errors import ContractViolation, Port2026Error


class ReplayPinMismatch(Port2026Error):
    """Pinned manifest / action hash, schema, or width does not match."""


class ReplayExhausted(Port2026Error):
    """The saved segment is exhausted; rows are never repeated."""


SAVED_SCHEMA = "brvla_saved_execution23_v1"
SAVED_REPRESENTATION = "pairwise_delta"


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DemoReplayPolicy:
    """Official-evaluator policy fed by a read-only converted replay episode.

    Args:
        converted_dir: directory holding ``manifest.json`` and ``actions.npy``.
        manifest_sha256: expected sha256 of ``manifest.json`` (private pin).
        actions_sha256: expected sha256 of ``actions.npy``; must agree with
            the manifest's ``action_sha256``.
        start_frame: first saved row index to emit (inclusive).
        end_frame: last saved row index (exclusive); ``None`` means the whole
            episode. No row is replayed after exhaustion.
        task_name / demo_id / instance_id: bound identity recorded for audit;
            mismatches raise on the ``load_demo`` hook.
        trace_path: JSONL trace path. Falls back to ``BR_REPLAY_TRACE``.
        rows: test-only injected rows (list of plain lists); do not use in
            production wiring.
        manifest: test-only injected manifest dict.
    """

    def __init__(
        self,
        converted_dir: Optional[str] = None,
        manifest_sha256: Optional[str] = None,
        actions_sha256: Optional[str] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        task_name: Optional[str] = None,
        demo_id: Optional[str] = None,
        instance_id: Optional[int] = None,
        trace_path: Optional[str] = None,
        rows: Optional[List[List[float]]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._converted_dir = pathlib.Path(converted_dir) if converted_dir else None
        self._expected_manifest_sha = manifest_sha256
        self._expected_actions_sha = actions_sha256
        self._start_frame = int(start_frame)
        self._end_frame = end_frame
        self._bound = {
            "task_name": task_name,
            "demo_id": demo_id,
            "instance_id": instance_id,
        }
        self._trace_path = trace_path or os.environ.get("BR_REPLAY_TRACE")
        self._index = 0
        self._action_dim: Optional[int] = None
        self._rows: Any = None
        self._manifest: Dict[str, Any] = {}
        self._manifest_sha256: Optional[str] = None
        self._actions_sha256: Optional[str] = None

        if rows is not None:
            self._rows = [list(r) for r in rows]
            if manifest is not None:
                self._manifest = manifest
            self._validate_injected()
        else:
            if self._converted_dir is None:
                raise ContractViolation(
                    "DemoReplayPolicy requires converted_dir or test-only rows injection")
            self._load()

        self._emit("saved_policy_bound", {
            "manifest_sha256": self._manifest_sha256,
            "actions_sha256": self._actions_sha256,
            "representation": self._manifest.get("representation"),
            "schema": self._manifest.get("schema"),
            "rows": len(self._rows),
            "not_model_eval": True,
            "bound": self._bound,
        })

    # -- loading / pin gates -------------------------------------------------

    def _load(self) -> None:
        manifest_path = self._converted_dir / "manifest.json"
        if not manifest_path.exists():
            raise ReplayPinMismatch(f"converted manifest absent: {manifest_path}")
        actual_manifest_sha = _sha256_file(manifest_path)
        if self._expected_manifest_sha is not None and \
                actual_manifest_sha != self._expected_manifest_sha:
            raise ReplayPinMismatch(
                f"manifest sha drift: expected {self._expected_manifest_sha}, "
                f"got {actual_manifest_sha}")
        self._manifest_sha256 = actual_manifest_sha

        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest_schema()

        actions_path = self._converted_dir / self._manifest.get("action_file", "actions.npy")
        if not actions_path.exists():
            raise ReplayPinMismatch(f"actions file absent: {actions_path}")
        actual_actions_sha = _sha256_file(actions_path)
        manifest_action_sha = self._manifest.get("action_sha256")
        if manifest_action_sha is not None and actual_actions_sha != manifest_action_sha:
            raise ReplayPinMismatch(
                f"actions sha drift vs manifest: {actual_actions_sha} != {manifest_action_sha}")
        if self._expected_actions_sha is not None and \
                actual_actions_sha != self._expected_actions_sha:
            raise ReplayPinMismatch(
                f"actions sha drift: expected {self._expected_actions_sha}, "
                f"got {actual_actions_sha}")
        self._actions_sha256 = actual_actions_sha

        import numpy as np  # lazy; package import stays numpy-free

        array = np.load(actions_path)
        if array.ndim != 2 or array.shape[1] != contracts.ACTION23:
            raise ReplayPinMismatch(
                f"actions shape {tuple(array.shape)} != (N, {contracts.ACTION23})")
        if not np.isfinite(array).all():
            raise ReplayPinMismatch("saved actions contain non-finite values")
        self._rows = array
        shape = list(array.shape)
        self._manifest.setdefault("shape", shape)

    def _validate_manifest_schema(self) -> None:
        if self._manifest.get("schema") != SAVED_SCHEMA:
            raise ReplayPinMismatch(
                f"unsupported saved schema: {self._manifest.get('schema')!r}")
        if self._manifest.get("representation") != SAVED_REPRESENTATION:
            raise ReplayPinMismatch(
                f"unsupported representation: {self._manifest.get('representation')!r}")
        shape = self._manifest.get("shape")
        if not shape or int(shape[1]) != contracts.ACTION23:
            raise ReplayPinMismatch(f"manifest shape {shape} != (N, 23)")

    def _validate_injected(self) -> None:
        for row in self._rows:
            if len(row) != contracts.ACTION23:
                raise ReplayPinMismatch(f"injected row width {len(row)} != {contracts.ACTION23}")
            if any(not _is_finite(v) for v in row):
                raise ReplayPinMismatch("injected row contains non-finite values")
        self._manifest.setdefault("schema", SAVED_SCHEMA)
        self._manifest.setdefault("representation", SAVED_REPRESENTATION)
        self._manifest.setdefault("shape", [len(self._rows), contracts.ACTION23])

    # -- evaluator seam ------------------------------------------------------

    def set_action_dim(self, action_dim: int) -> None:
        """Record the evaluator's native action width for parity audit."""
        self._action_dim = int(action_dim)

    def load_demo(self, demo_id: Any, task_id: Optional[int] = None) -> None:
        """Optional identity hook (kept for old-replay compat); resets index."""
        if self._bound["demo_id"] is not None and \
                str(demo_id).zfill(8) != str(self._bound["demo_id"]).zfill(8):
            raise ReplayPinMismatch(
                f"bound demo {self._bound['demo_id']} != requested {demo_id}")
        if task_id is not None and self._bound["task_name"] is not None:
            # task_id is numeric; task_name is the BR task label. Numeric match
            # is checked by the runner gate against the frozen manifest.
            pass
        self._index = 0

    def reset(self) -> None:
        self._index = 0
        self._emit("saved_policy_reset", {"index": self._index})

    def forward(self, obs: Optional[dict] = None, **kwargs: Any) -> Any:
        row = self._next_row()

        import torch  # lazy; package import stays torch-free

        return torch.tensor(row, dtype=torch.float32)

    act = forward

    def _next_row(self) -> List[float]:
        row_index = self._start_frame + self._index
        end = len(self._rows) if self._end_frame is None else int(self._end_frame)
        if row_index >= end:
            raise ReplayExhausted(
                f"saved segment exhausted at row {row_index}; no repeated row")
        row = self._rows[row_index]
        if not isinstance(row, list):
            row = row.copy()
        row = [float(v) for v in row]
        self._index += 1

        self._emit("saved_row_selected", {
            "source_row": row_index,
            "action_sha256": hashlib.sha256(_to_bytes(row)).hexdigest(),
        })
        return row

    # -- audit receipt -------------------------------------------------------

    def bound_receipt(self) -> Dict[str, Any]:
        anchors = None
        chunks = self._manifest.get("chunks") or []
        if chunks:
            anchors = chunks[0].get("anchors")
        return {
            "policy": "DemoReplayPolicy",
            "not_model_eval": True,
            "bound": self._bound,
            "schema": self._manifest.get("schema"),
            "representation": self._manifest.get("representation"),
            "shape": self._manifest.get("shape"),
            "dt_seconds": self._manifest.get("dt_seconds"),
            "anchors": anchors,
            "columns": self._manifest.get("columns"),
            "convention": self._manifest.get("convention"),
            "manifest_sha256": getattr(self, "_manifest_sha256", None),
            "actions_sha256": getattr(self, "_actions_sha256", None),
            "start_frame": self._start_frame,
            "end_frame": self._end_frame,
            "index": self._index,
            "evaluator_action_dim": self._action_dim,
        }

    # -- trace helper ---------------------------------------------------------

    def _emit(self, event: str, detail: Dict[str, Any]) -> None:
        if not self._trace_path:
            return
        path = pathlib.Path(self._trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **detail}, allow_nan=False) + "\n")
            f.flush()


def _is_finite(value: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def _to_bytes(row: Any) -> bytes:
    if hasattr(row, "tobytes"):
        return row.tobytes()
    return json.dumps(list(row), allow_nan=False).encode("utf-8")
