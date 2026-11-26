"""Source-to-live object identity matching for the MoMa three-can PnP lane.

Boundary (canary attempt3 evidence, gap G1):

The a28 source replay names objects by source-sim numeric suffix
(``can_of_soda_115`` / ``can_of_soda_114`` / ``can_of_soda_113`` /
``trash_can_116``). The 2026 TRO ``object_scopes`` keys are BDDL class instance
names (``can__of__soda.n.01_1`` ..., ``ashcan.n.01_1``). A class-only three-can
alignment is **never** accepted as the same initial state: identity must be
established by an exact name / prim-path match, or the candidate set stays
``unconfirmed`` with its pose residuals recorded for a later live probe.

Source-side identity persistence in episode ``episode_00010010.hdf5``:

- ``init_metadata`` group is **empty**; per-object uid / category / model are
  not persisted in this episode.
- The npz stores only a 4x4 world pose trajectory per named object
  (``object_can_of_soda_*`` / ``object_trash_can_116``).

Therefore source ``uid`` / ``category`` / ``model`` default to ``UNKNOWN`` here;
they are not reconstructed from the object name string.

Initial-state reconstruction contract (not equivalent to a single-frame load):

a28 restores the initial state by, in order, ``scene.restore`` -> apply
``init_metadata`` attributes -> ``env.reset()`` -> replay ``state[0..818]``
sequentially via ``load_state(serialized=True)`` -> invalidate pose cache ->
render. Loading only ``state[818]`` probes serialized-state API compatibility;
it does **not** reproduce the sequential replay and must not be claimed as a
matching initial state.

This module is numpy-only and sim-free: the live object is duck-typed, every
read is failure-isolated to an UNKNOWN record, and no omnigibson/torch import
happens at module load.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Source name prefix -> role + live tokens matched against lowercased
# live name / category / prim_path. Tokens are substrings, not regexes.
SOURCE_ROLE_LEXICON: Dict[str, Dict[str, Any]] = {
    "can_of_soda": {
        "role": "can",
        "live_tokens": ("can__of__soda", "can_of_soda", "soda", "soda_can"),
    },
    "trash_can": {
        "role": "bucket",
        "live_tokens": ("ashcan", "trash_can", "trashcan", "waste"),
    },
}

# Verdicts that count as an identity confirmation. Everything else is a
# candidate or a refusal, never a same-object claim.
IDENTITY_CONFIRMED_VERDICTS = frozenset({"name_exact", "prim_exact"})

_NAME_SUFFIX_RE = re.compile(r"^(.*?)(?:_(\d+))?$")


def _clean(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"status": "UNKNOWN", "error": "nonfinite_value"}
    return value


def parse_source_name(source_name: str) -> Dict[str, Any]:
    """Split a source object name into prefix and numeric suffix.

    Returns ``{"role": str|None, "prefix": str, "suffix": str|None}``.
    """
    match = _NAME_SUFFIX_RE.match(source_name)
    prefix = match.group(1) if match else source_name
    suffix = match.group(2) if match else None
    lex = SOURCE_ROLE_LEXICON.get(prefix)
    return {"role": lex["role"] if lex else None, "prefix": prefix, "suffix": suffix}


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Scalar-last quaternion ``(x, y, z, w)`` -> 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    if not np.isfinite(q).all():
        raise ValueError("quat_nonfinite")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_from_pos_quat(pos: Any, quat: Any) -> np.ndarray:
    """Build a 4x4 world pose from a position and scalar-last quaternion."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    mat = quat_to_mat(quat)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = mat
    pose[:3, 3] = p
    return pose


def _as_pose4x4(pose: Any) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise ValueError("invalid_pose_4x4")
    return arr


def pose_residual(source_pose: Any, live_pose: Any) -> Dict[str, Any]:
    """Translation (m) and geodesic rotation (deg) between two 4x4 poses."""
    src = _as_pose4x4(source_pose)
    live = _as_pose4x4(live_pose)
    trans = float(np.linalg.norm(live[:3, 3] - src[:3, 3]))
    rot = live[:3, :3].T @ src[:3, :3]
    cos_angle = (np.trace(rot) - 1.0) / 2.0
    cos_angle = min(1.0, max(-1.0, cos_angle))
    rot_deg = float(math.degrees(math.acos(cos_angle)))
    return {"translation_m": trans, "rotation_deg": rot_deg, "finite": True}


@dataclass
class SourceObjectRef:
    """A source-replay object reference at the initial-state frame (index 818)."""

    role: str
    source_name: str
    source_pose: Any
    source_uid: Optional[str] = None
    category: Optional[str] = None
    model: Optional[str] = None
    source_index: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def record(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "source_name": self.source_name,
            "source_uid": self.source_uid if self.source_uid is not None else "UNKNOWN",
            "category": self.category if self.category is not None else "UNKNOWN",
            "model": self.model if self.model is not None else "UNKNOWN",
            "source_index": self.source_index,
            "source_pose": _clean(np.asarray(self.source_pose, dtype=float).tolist()),
        }


def live_snapshot(obj: Any) -> Dict[str, Any]:
    """Read identity fields from a duck-typed live object; failure -> UNKNOWN.

    Expected post1 surface: ``name``, ``prim_path``, ``relative_prim_path``,
    ``category``, ``_uuid``, ``_usd_path``, ``get_position_orientation()``.
    Missing fields do not raise; they are recorded UNKNOWN.
    """
    snap: Dict[str, Any] = {"object_ref": "live_duck_typed"}

    def read(name: str, fn: Callable[[], Any]) -> None:
        try:
            snap[name] = _clean(fn())
        except Exception as exc:  # noqa: BLE001 - identity read failure stays UNKNOWN
            snap[name] = {"status": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}

    read("name", lambda: obj.name)
    read("prim_path", lambda: obj.prim_path)
    read("relative_prim_path", lambda: getattr(obj, "relative_prim_path", None))
    read("category", lambda: getattr(obj, "category", None))
    read("uuid", lambda: getattr(obj, "_uuid", getattr(obj, "uuid", None)))
    read("model", lambda: getattr(obj, "_usd_path", getattr(obj, "usd_path", None)))

    def pose() -> List[List[float]]:
        pos, quat = obj.get_position_orientation()
        return pose_from_pos_quat(pos, quat).tolist()

    read("pose", pose)
    return snap


def _tokens_match(role: str, snap: Dict[str, Any], lexicon_tokens: Tuple[str, ...]) -> bool:
    haystack_parts: List[str] = []
    for key in ("name", "category", "prim_path", "relative_prim_path"):
        val = snap.get(key)
        if isinstance(val, str):
            haystack_parts.append(val.lower())
    haystack = " ".join(haystack_parts)
    return any(tok in haystack for tok in lexicon_tokens)


def _name_or_prim_exact(source_name: str, snap: Dict[str, Any]) -> Optional[str]:
    live_name = snap.get("name")
    if isinstance(live_name, str) and live_name == source_name:
        return "name_exact"
    prim = snap.get("prim_path")
    if isinstance(prim, str) and prim.rsplit("/", 1)[-1] == source_name:
        return "prim_exact"
    return None


def match_objects(
    refs: List[SourceObjectRef],
    live_objects: List[Any],
    pose_tol_translation_m: float = 0.05,
    pose_tol_rotation_deg: float = 5.0,
) -> Dict[str, Any]:
    """Match each source reference against live scope objects.

    Parameters
    ----------
    refs:
        Source references at the initial-state frame.
    live_objects:
        Live scope-value objects (duck-typed); ``live_snapshot`` reads them.
    pose_tol_*:
        Residual thresholds used only to label a weak ``role_pose`` candidate.
        A residual inside threshold still does **not** confirm identity when the
        scene is not the source replay scene.

    Returns a report keyed by source name. ``identity_confirmed`` is true only
    for an exact name/prim-path match; otherwise the matched object stays
    ``None`` and the candidate list / residuals are recorded.
    """
    snapshots = [live_snapshot(obj) for obj in live_objects]
    report: Dict[str, Any] = {"pose_tolerance": {
        "translation_m": pose_tol_translation_m,
        "rotation_deg": pose_tol_rotation_deg,
    }, "matches": {}}

    for ref in refs:
        lex = SOURCE_ROLE_LEXICON.get(parse_source_name(ref.source_name)["prefix"])
        tokens = tuple(lex["live_tokens"]) if lex else ()
        candidates: List[Dict[str, Any]] = []

        for snap in snapshots:
            role_match = bool(tokens) and _tokens_match(ref.role, snap, tokens)
            entry: Dict[str, Any] = {
                "name": snap.get("name"),
                "prim_path": snap.get("prim_path"),
                "category": snap.get("category"),
                "model": snap.get("model"),
                "uuid": snap.get("uuid"),
                "role_match": role_match,
            }
            exact = _name_or_prim_exact(ref.source_name, snap)
            entry["exact_match"] = exact

            residual: Dict[str, Any] = {"status": "UNKNOWN", "error": "pose_unavailable"}
            live_pose = snap.get("pose")
            if isinstance(live_pose, list):
                try:
                    residual = pose_residual(ref.source_pose, live_pose)
                    within = (residual["translation_m"] <= pose_tol_translation_m
                              and residual["rotation_deg"] <= pose_tol_rotation_deg)
                    residual["within_pose_tolerance"] = within
                except Exception as exc:  # noqa: BLE001
                    residual = {"status": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}
            entry["pose_residual"] = residual
            candidates.append(entry)

        role_candidates = [c for c in candidates if c["role_match"]]
        exact_candidates = [c for c in candidates if c["exact_match"]]

        if exact_candidates:
            chosen = exact_candidates[0]
            verdict = chosen["exact_match"]
            matched = chosen["name"]
            identity_confirmed = True
        elif not role_candidates:
            verdict = "no_role_match"
            chosen = None
            matched = None
            identity_confirmed = False
        else:
            pose_ok = [
                c for c in role_candidates
                if isinstance(c["pose_residual"], dict)
                and c["pose_residual"].get("within_pose_tolerance") is True
            ]
            if len(role_candidates) == 1 and pose_ok:
                verdict = "unconfirmed_role_pose"
            else:
                verdict = "unconfirmed_role_only"
            chosen = None
            matched = None
            identity_confirmed = False

        report["matches"][ref.source_name] = {
            "source": ref.record(),
            "verdict": verdict,
            "identity_confirmed": identity_confirmed,
            "matched_live_name": matched,
            "candidate_count": len(candidates),
            "role_candidate_count": len(role_candidates),
            "candidates": candidates,
        }

    report["identity_confirmed_count"] = sum(
        1 for m in report["matches"].values() if m["identity_confirmed"]
    )
    report["identity_confirmed"] = (
        report["identity_confirmed_count"] == len(refs) and len(refs) > 0
    )
    return report
