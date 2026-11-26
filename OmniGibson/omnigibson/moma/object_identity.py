"""Source-to-live object identity matching for the MoMa three-can PnP lane.

Identity semantics (corrected, post b13 review):

The BDDL instance id (``can__of__soda.n.01_1`` ..., ``ashcan.n.01_1``) is a
**role binding key** that is reused across episodes/scenes. Matching this key
to a live scope entry binds the role slot; it does **not** by itself prove the
same object, the same source scene, or the same initial state. A public_test
instance that happens to share the role names is not a same-episode match.

A confirmation (``identity_confirmed = True``) requires the full closure
correspondence:

1. role binding: exact BDDL instance / source name / prim-path match;
2. category match against the frozen source closure;
3. model id match against the frozen source closure;
4. asset-hash match when both sides expose one;
5. source scene closure match (scene model/instance, config attr sha,
   scene-file attr sha, dataset versions).

Category, model, asset-hash, or scene-closure mismatch is an explicit
``rejected_*`` verdict, never an unconfirmed candidate. A role-bound object
without a verified live scene closure stays ``role_bound_scene_unverified``.

Pose epochs are kept separate and are never blended:

- ``registry_root_link_pose``: baked scene initial pose from the source
  closure registry (pos + scalar-last ori).
- ``npz_pose_frame0`` / ``npz_pose_frame818``: full 4x4 SE3 from the decoded
  npz trajectory for that epoch frame.

``source_pose`` is a single-epoch complete SE3 (npz frame818 by default).
If orientation is missing for that epoch, ``source_pose`` is recorded UNKNOWN
instead of being mixed with another epoch's rotation.

Source fields that are NOT persisted in this episode stay ``UNKNOWN`` and are
not reconstructed from the name string: per-object ``source_uid`` and the
runtime ``prim_path`` (assigned at live load). The ``init_metadata`` group on
``demo_0`` is empty.

Initial-state reconstruction contract (not equivalent to a single-frame load):

a28 restores the initial state through the sealed ``DataPlaybackWrapper``
entry: ``create_env`` reads the hdf5 config/scene closure -> ``scene.restore``
-> apply ``init_metadata`` -> ``env.reset()`` -> replay ``state[0..818]`` in
order via ``load_state(serialized=True)`` -> invalidate pose cache -> render.
Loading only ``state[818]`` probes serialized-state API compatibility; it
does **not** reproduce the sequential replay and must not be claimed as a
matching initial state.

This module is numpy-only and sim-free: the live object is duck-typed, every
read is failure-isolated to an UNKNOWN record, and no omnigibson/torch import
happens at module load.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Source category prefix -> role + live tokens matched against lowercased
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
# rejected role-slot candidate or a refusal, never a same-object claim.
IDENTITY_CONFIRMED_VERDICTS = frozenset({"closure_confirmed"})

# Explicit rejection verdicts: closure evidence disagrees with the source.
REJECTED_VERDICTS = frozenset(
    {"rejected_category", "rejected_model", "rejected_asset_hash",
     "rejected_scene_closure"})

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


def _closure_eq(src: Any, live: Any) -> Optional[bool]:
    """Three-state equality: True/False, None when either side is UNKNOWN."""
    if src is None or live is None:
        return None
    if isinstance(src, str) and src == "UNKNOWN":
        return None
    if isinstance(live, dict) and live.get("status") == "UNKNOWN":
        return None
    if isinstance(live, str) and live == "UNKNOWN":
        return None
    return src == live


@dataclass
class SourceSceneRef:
    """Frozen source scene/episode closure fingerprint."""

    scene_model: Optional[str] = None
    scene_instance: Optional[str] = None
    config_attr_sha256: Optional[str] = None
    scene_file_attr_sha256: Optional[str] = None
    versions: Optional[Dict[str, Any]] = None
    episode: Optional[str] = None

    def record(self) -> Dict[str, Any]:
        return {
            "scene_model": self.scene_model or "UNKNOWN",
            "scene_instance": self.scene_instance or "UNKNOWN",
            "config_attr_sha256": self.config_attr_sha256 or "UNKNOWN",
            "scene_file_attr_sha256": self.scene_file_attr_sha256 or "UNKNOWN",
            "versions": _clean(self.versions) if self.versions else "UNKNOWN",
            "episode": self.episode or "UNKNOWN",
        }

    def match(self, live_closure: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare closure fields; UNKNOWN when the live closure is absent."""
        if not isinstance(live_closure, dict):
            return {"scene_match": None, "reason": "live_scene_closure_absent",
                    "mismatch": []}
        pairs = [
            ("scene_model", self.scene_model, live_closure.get("scene_model")),
            ("scene_instance", self.scene_instance, live_closure.get("scene_instance")),
            ("config_attr_sha256", self.config_attr_sha256,
             live_closure.get("config_attr_sha256")),
            ("scene_file_attr_sha256", self.scene_file_attr_sha256,
             live_closure.get("scene_file_attr_sha256")),
        ]
        mismatch: List[str] = []
        for field_name, src, live in pairs:
            eq = _closure_eq(src, live)
            if eq is False:
                mismatch.append(field_name)
        if mismatch:
            return {"scene_match": False, "mismatch": mismatch}
        if any(_closure_eq(s, l) is None for _, s, l in pairs):
            return {"scene_match": None, "reason": "closure_field_unverified",
                    "mismatch": []}
        return {"scene_match": True, "mismatch": []}


@dataclass
class SourceObjectRef:
    """A source-replay object reference anchored to the sealed episode closure.

    ``source_pose`` is a single-epoch complete SE3. Cross-epoch pose blending
    is prohibited; other epochs are kept in ``extras`` as separate records.
    """

    role: str
    source_name: str
    source_pose: Any
    bddl_instance: Optional[str] = None
    source_uid: Optional[str] = None
    category: Optional[str] = None
    model: Optional[str] = None
    expected_file_hash: Optional[str] = None
    in_rooms: Optional[List[str]] = None
    scene: Optional[SourceSceneRef] = None
    source_index: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def record(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "source_name": self.source_name,
            "bddl_instance": self.bddl_instance if self.bddl_instance else "UNKNOWN",
            "source_uid": self.source_uid if self.source_uid is not None else "UNKNOWN",
            "category": self.category if self.category is not None else "UNKNOWN",
            "model": self.model if self.model is not None else "UNKNOWN",
            "expected_file_hash": self.expected_file_hash or "UNKNOWN",
            "in_rooms": self.in_rooms if self.in_rooms else "UNKNOWN",
            "source_index": self.source_index,
            "source_pose": _clean(self.source_pose),
            "scene": self.scene.record() if self.scene else "UNKNOWN",
        }


def refs_from_anchor(anchor_rows: List[Dict[str, Any]],
                     scene: Optional[SourceSceneRef] = None,
                     pose_epoch: str = "npz_pose_frame818") -> List[SourceObjectRef]:
    """Build refs from the probe 116 anchor table.

    Parameters
    ----------
    anchor_rows:
        Rows from ``moma_source_anchor_table_20261126.json`` ->
        ``anchor.anchor_rows``.
    scene:
        Frozen source scene closure fingerprint shared by all rows.
    pose_epoch:
        Which single-epoch full SE3 to keep as ``source_pose``. Default
        ``npz_pose_frame818``. The other epoch and the registry baked pose are
        carried in ``extras`` untouched. If the chosen epoch pose is not a
        complete 4x4 finite SE3, ``source_pose`` is an UNKNOWN record rather
        than a cross-epoch blend.
    """
    allowed = {"npz_pose_frame0", "npz_pose_frame818"}
    if pose_epoch not in allowed:
        raise ValueError(f"unknown_pose_epoch: {pose_epoch}")

    refs: List[SourceObjectRef] = []
    for row in anchor_rows:
        name = row["source_name"]
        prefix = parse_source_name(name)["prefix"]
        lex = SOURCE_ROLE_LEXICON.get(prefix)
        role = row.get("category") or (lex["role"] if lex else None)

        pose_raw = row.get(pose_epoch)
        source_pose: Any
        try:
            pose_arr = _as_pose4x4(pose_raw)
            source_pose = pose_arr.tolist()
        except (ValueError, TypeError, IndexError):
            source_pose = {"status": "UNKNOWN",
                           "error": "pose_epoch_incomplete_no_cross_epoch_blend"}

        extras: Dict[str, Any] = {
            "registry_root_link_pose": row.get("registry_root_link_pose"),
            "npz_pose_frame0": row.get("npz_pose_frame0"),
            "npz_pose_frame818": row.get("npz_pose_frame818"),
            "pose_epoch": pose_epoch,
        }
        refs.append(SourceObjectRef(
            role=role or "unknown",
            source_name=name,
            source_pose=source_pose,
            bddl_instance=row.get("bddl_instance"),
            category=row.get("category"),
            model=row.get("model"),
            expected_file_hash=row.get("expected_file_hash"),
            in_rooms=row.get("in_rooms"),
            scene=scene,
            extras=extras,
        ))
    return refs


def live_snapshot(obj: Any, scope_key: Optional[str] = None) -> Dict[str, Any]:
    """Read identity fields from a duck-typed live object; failure -> UNKNOWN.

    Expected post1 surface: ``scope_key`` (task.object_scopes entry), ``name``,
    ``prim_path``, ``relative_prim_path``, ``category``, ``model``,
    ``in_rooms``, ``_expected_file_hash``, ``usd_path``, ``_uuid``,
    ``get_position_orientation()``. Missing fields do not raise; they are
    recorded UNKNOWN.
    """
    snap: Dict[str, Any] = {"object_ref": "live_duck_typed", "scope_key": scope_key}

    def read(name: str, fn: Callable[[], Any]) -> None:
        try:
            snap[name] = _clean(fn())
        except Exception as exc:  # noqa: BLE001 - identity read failure stays UNKNOWN
            snap[name] = {"status": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}

    read("name", lambda: obj.name)
    read("prim_path", lambda: obj.prim_path)
    read("relative_prim_path", lambda: getattr(obj, "relative_prim_path", None))
    read("category", lambda: getattr(obj, "category", None))
    read("model", lambda: getattr(obj, "model", getattr(obj, "_model", None)))
    read("in_rooms", lambda: getattr(obj, "in_rooms", None))
    read("expected_file_hash",
         lambda: getattr(obj, "_expected_file_hash", getattr(obj, "expected_file_hash", None)))
    read("usd_path", lambda: getattr(obj, "usd_path", getattr(obj, "_usd_path", None)))
    read("uuid", lambda: getattr(obj, "_uuid", getattr(obj, "uuid", None)))

    def pose() -> List[List[float]]:
        pos, quat = obj.get_position_orientation()
        return pose_from_pos_quat(pos, quat).tolist()

    read("pose", pose)
    return snap


def _tokens_match(role: str, snap: Dict[str, Any], lexicon_tokens: Tuple[str, ...]) -> bool:
    haystack_parts: List[str] = []
    for key in ("name", "category", "prim_path", "relative_prim_path", "scope_key"):
        val = snap.get(key)
        if isinstance(val, str):
            haystack_parts.append(val.lower())
    haystack = " ".join(haystack_parts)
    return any(tok in haystack for tok in lexicon_tokens)


def _role_bound(ref: SourceObjectRef, snap: Dict[str, Any]) -> Optional[str]:
    """Exact role-slot binding: instance key, then source name, then prim."""
    if ref.bddl_instance:
        for key in ("scope_key", "name"):
            val = snap.get(key)
            if isinstance(val, str) and val == ref.bddl_instance:
                return "instance_exact"
    live_name = snap.get("name")
    if isinstance(live_name, str) and live_name == ref.source_name:
        return "name_exact"
    prim = snap.get("prim_path")
    if isinstance(prim, str) and prim.rsplit("/", 1)[-1] == ref.source_name:
        return "prim_exact"
    return None


def _corroboration(ref: SourceObjectRef, snap: Dict[str, Any]) -> Dict[str, Any]:
    """Category/model/asset-hash/room closure evidence per candidate."""
    out: Dict[str, Any] = {
        "category_match": _closure_eq(ref.category, snap.get("category")),
        "model_match": _closure_eq(ref.model, snap.get("model")),
        "asset_hash_match": _closure_eq(ref.expected_file_hash,
                                         snap.get("expected_file_hash")),
    }
    if ref.in_rooms:
        live_rooms = snap.get("in_rooms")
        out["room_overlap"] = bool(
            isinstance(live_rooms, list) and set(ref.in_rooms) & set(map(str, live_rooms)))
    else:
        out["room_overlap"] = None
    return out


def _verdict_for(ref: SourceObjectRef, bound: Optional[str],
                 corroboration: Dict[str, Any],
                 scene_result: Dict[str, Any]) -> str:
    if bound is None:
        return "no_role_match"
    if corroboration["category_match"] is False:
        return "rejected_category"
    if corroboration["model_match"] is False:
        return "rejected_model"
    if corroboration["asset_hash_match"] is False:
        return "rejected_asset_hash"
    if scene_result.get("scene_match") is False:
        return "rejected_scene_closure"
    if scene_result.get("scene_match") is None:
        return "role_bound_scene_unverified"
    if corroboration["category_match"] and corroboration["model_match"]:
        return "closure_confirmed"
    return "role_bound_scene_unverified"


def match_objects(
    refs: List[SourceObjectRef],
    live_objects: List[Any],
    pose_tol_translation_m: float = 0.05,
    pose_tol_rotation_deg: float = 5.0,
    scope_keys: Optional[List[Optional[str]]] = None,
    live_scene_closure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Match each source reference against live scope objects.

    Parameters
    ----------
    refs:
        Source references anchored to the sealed episode closure.
    live_objects:
        Live scope-value objects (duck-typed); ``live_snapshot`` reads them.
    scope_keys:
        Optional ``task.object_scopes`` keys aligned positionally with
        ``live_objects``.
    live_scene_closure:
        Live scene fingerprint (scene model/instance, config/scene attr shas,
        versions). Absent -> every role-bound candidate stays unverified;
        closure confirmation is impossible.
    pose_tol_*:
        Residual thresholds used only to label a weak role candidate. A
        residual inside tolerance across different scenes does not confirm
        identity.

    Returns a report keyed by source name. ``identity_confirmed`` is true only
    for ``closure_confirmed``; rejected/unverified candidates keep ``None``
    matched object and record their residuals.
    """
    keys = scope_keys or [None] * len(live_objects)
    snapshots = [live_snapshot(obj, key) for obj, key in zip(live_objects, keys)]
    report: Dict[str, Any] = {"pose_tolerance": {
        "translation_m": pose_tol_translation_m,
        "rotation_deg": pose_tol_rotation_deg,
    }, "live_scene_closure": live_scene_closure, "matches": {}}

    for ref in refs:
        lex = SOURCE_ROLE_LEXICON.get(parse_source_name(ref.source_name)["prefix"])
        tokens = tuple(lex["live_tokens"]) if lex else ()
        scene_result = ref.scene.match(live_scene_closure) if ref.scene else \
            {"scene_match": None, "reason": "source_scene_ref_absent", "mismatch": []}
        candidates: List[Dict[str, Any]] = []

        for snap in snapshots:
            role_match = bool(tokens) and _tokens_match(ref.role, snap, tokens)
            bound = _role_bound(ref, snap)
            corroboration = _corroboration(ref, snap)
            entry: Dict[str, Any] = {
                "scope_key": snap.get("scope_key"),
                "name": snap.get("name"),
                "prim_path": snap.get("prim_path"),
                "category": snap.get("category"),
                "model": snap.get("model"),
                "in_rooms": snap.get("in_rooms"),
                "expected_file_hash": snap.get("expected_file_hash"),
                "uuid": snap.get("uuid"),
                "role_match": role_match,
                "role_binding": bound,
                "corroboration": corroboration,
                "scene_match": scene_result,
            }

            residual: Dict[str, Any] = {"status": "UNKNOWN", "error": "pose_unavailable"}
            live_pose = snap.get("pose")
            source_pose = ref.source_pose
            if isinstance(live_pose, list) and not isinstance(source_pose, dict):
                try:
                    residual = pose_residual(source_pose, live_pose)
                    within = (residual["translation_m"] <= pose_tol_translation_m
                              and residual["rotation_deg"] <= pose_tol_rotation_deg)
                    residual["within_pose_tolerance"] = within
                except Exception as exc:  # noqa: BLE001
                    residual = {"status": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}
            entry["pose_residual"] = residual
            entry["verdict"] = _verdict_for(ref, bound, corroboration, scene_result)
            candidates.append(entry)

        confirmed = [c for c in candidates if c["verdict"] == "closure_confirmed"]
        rejected = [c for c in candidates if c["verdict"] in REJECTED_VERDICTS]
        role_bound = [c for c in candidates if c["role_binding"]]
        role_candidates = [c for c in candidates if c["role_match"]]

        if confirmed:
            chosen = confirmed[0]
            verdict = "closure_confirmed"
            matched = chosen["scope_key"] or chosen["name"]
            identity_confirmed = True
        else:
            chosen = None
            matched = None
            identity_confirmed = False
            if rejected:
                verdict = rejected[0]["verdict"]
            elif role_bound:
                verdict = "role_bound_scene_unverified"
            elif not role_candidates:
                verdict = "no_role_match"
            else:
                pose_ok = [
                    c for c in role_candidates
                    if isinstance(c["pose_residual"], dict)
                    and c["pose_residual"].get("within_pose_tolerance") is True
                ]
                verdict = ("unconfirmed_role_pose"
                           if len(role_candidates) == 1 and pose_ok
                           else "unconfirmed_role_only")

        report["matches"][ref.source_name] = {
            "source": ref.record(),
            "verdict": verdict,
            "identity_confirmed": identity_confirmed,
            "matched_live_name": matched,
            "candidate_count": len(candidates),
            "role_candidate_count": len(role_candidates),
            "rejected_count": len(rejected),
            "candidates": candidates,
        }

    report["identity_confirmed_count"] = sum(
        1 for m in report["matches"].values() if m["identity_confirmed"]
    )
    report["identity_confirmed"] = (
        report["identity_confirmed_count"] == len(refs) and len(refs) > 0
    )
    return report
