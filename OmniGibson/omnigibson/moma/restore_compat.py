"""Source-compatible restore bootstrap / reader contract (CPU, sim-free).

This module is the CPU seam for migrating the sealed a28 source episode state
onto the post1 managed scene. It does **not** launch a simulator, does not
import omnigibson/torch, and never mutates the source hdf5 group.

State packing fact (verified read-only against post1 `bd049de3…60fa20c108`):

- The serialized flat row is ``[n_objs, <uuid_int, obj_state> × n_objs]``.
- `SerializableRegistry.deserialize` resolves each segment by
  ``hash_key="uuid"`` and asserts the object exists; it advances the index by
  the per-object ``deserialized_items`` boundary.
- The uuid is deterministic from the object name:
  ``get_uuid(name) = int(float32(md5(name) mod 10**8))`` (8 digits).

Consequences enforced here:

1. **Name preservation** keeps the uuid identical under the original formula.
   If the target keeps the source numeric name, segments align without hand
   editing the serialized payload. The same-name uuid equality is checked by
   recomputing the original formula, not by guessing.
2. **Renamed / cross-generation target** requires an explicit source-uuid ->
   target-uuid/name remap evidenced by the closure. An unmapped required
   segment blocks the **entire** restore; a partial state set is never a
   success.
3. Role binding or class/name coincidence alone is not the same initial
   state; closure confirmation still follows `object_identity` verdicts.
4. Source ``config``/``scene_file``/version SHAs are compared against the
   frozen source pin only. They are not expected to equal the post1 target
   scene, so an exact-sha target match is not a blocker.

The runtime sealed entry order (DataPlaybackWrapper / create_env) is recorded
in :class:`BootstrapOrder`; this CPU layer only produces the mapped restore
plan and its fail-closed verdict.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .object_identity import (
    IDENTITY_CONFIRMED_VERDICTS,
    REJECTED_VERDICTS,
    SourceObjectRef,
    SourceSceneRef,
    match_objects,
)

# Post1 uuid length: 8 digits, float32 rounded (see python_utils.get_uuid).
_UUID_N_DIGITS = 8
_UUID_MOD = 10 ** _UUID_N_DIGITS

#: Restore steps taken from the sealed create_env / bridge restore order.
BOOTSTRAP_STEPS: Tuple[str, ...] = (
    "create_env_read_attrs_config_scene_file",
    "wrap_data_playback",
    "load_observation_space",
    "scene_restore_update_initial_file",
    "apply_init_metadata",
    "env_reset",
    "replay_state_0_to_818_serialized",
    "pose_invalidate_render",
)

#: Physics/observation checkpoints preserved through the mapped restore.
FRAME_CHECKPOINTS: Tuple[str, ...] = ("frame0", "grasp", "physics30", "physics120")

#: Any blocked required segment makes the whole restore BLOCKED. Partial
#: state sets are never reported as a successful initial state.
PARTIAL_SUCCESS_ALLOWED = False

# Failure / block reason codes used by the restore plan.
REASON_SOURCE_PIN_MISMATCH = "source_pin_mismatch"
REASON_ROLE_ONLY_NOT_IDENTITY = "role_only_not_identity"
REASON_REJECTED = "closure_rejected"
REASON_SCENE_UNVERIFIED = "scene_closure_unverified"
REASON_UUID_UNALIGNED_RENAMED = "uuid_unaligned_name_renamed"
REASON_REQUIRED_SEGMENT_UNMAPPED = "required_segment_unmapped"
REASON_FORMULA_ERROR = "uuid_formula_error"


def compute_uuid(name: str) -> int:
    """Replicate post1 ``get_uuid(name, n_digits=8, deterministic=True)``.

    Original formula (``omnigibson/utils/python_utils.py``):
    ``val = int(md5(name.encode()).hexdigest(), 16)``; uuid =
    ``int(tensor(val % 10**8, dtype=float32).item())``. Numpy float32 is the
    same IEEE round as the torch scalar. The formula is recomputed from the
    original source string; no uuid is guessed or taken from a name suffix.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("uuid_requires_nonempty_name")
    val = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    return int(np.float32(val % _UUID_MOD).item())


def _clean(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


@dataclass
class SourcePin:
    """Frozen source episode fingerprint. Target scene SHAs are NOT compared
    against this pin; it identifies the sealed source only."""

    demo_sha256: Optional[str] = None
    config_attr_sha256: Optional[str] = None
    scene_file_attr_sha256: Optional[str] = None
    episode: Optional[str] = None
    scene_model: Optional[str] = None
    scene_instance: Optional[str] = None
    versions: Optional[Dict[str, Any]] = None
    state_schema: Optional[Dict[str, Any]] = None

    def record(self) -> Dict[str, Any]:
        return {
            "demo_sha256": self.demo_sha256 or "UNKNOWN",
            "config_attr_sha256": self.config_attr_sha256 or "UNKNOWN",
            "scene_file_attr_sha256": self.scene_file_attr_sha256 or "UNKNOWN",
            "episode": self.episode or "UNKNOWN",
            "scene_model": self.scene_model or "UNKNOWN",
            "scene_instance": self.scene_instance or "UNKNOWN",
            "versions": _clean(self.versions) if self.versions else "UNKNOWN",
            "state_schema": _clean(self.state_schema) if self.state_schema else "UNKNOWN",
        }

    def matches(self, live_source_attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Three-state source-pin check. Absent evidence -> None, not a pass."""
        if not isinstance(live_source_attrs, dict):
            return {"pin_match": None, "reason": "source_attrs_absent", "mismatch": []}
        pairs = [
            ("demo_sha256", self.demo_sha256, live_source_attrs.get("demo_sha256")),
            ("config_attr_sha256", self.config_attr_sha256,
             live_source_attrs.get("config_attr_sha256")),
            ("scene_file_attr_sha256", self.scene_file_attr_sha256,
             live_source_attrs.get("scene_file_attr_sha256")),
            ("episode", self.episode, live_source_attrs.get("episode")),
        ]
        mismatch: List[str] = []
        unverified: List[str] = []
        for field_name, src, live in pairs:
            if src is None or live is None:
                unverified.append(field_name)
                continue
            if str(src) != str(live):
                mismatch.append(field_name)
        if mismatch:
            return {"pin_match": False, "reason": REASON_SOURCE_PIN_MISMATCH,
                    "mismatch": mismatch}
        if unverified:
            return {"pin_match": None, "reason": "pin_field_unverified",
                    "mismatch": []}
        return {"pin_match": True, "mismatch": []}


@dataclass
class TargetMapEntry:
    """One source-object -> target-object restore mapping.

    ``uuid_mode``:

    - ``name_preserved``: target keeps the source numeric name; uuids align by
      recomputing the original formula.
    - ``explicit_remap``: target name differs; the post1 target uuid must be
      evidenced at runtime and recorded here. Without live evidence the entry
      stays unverified and blocks restore.
    """

    source_name: str
    bddl_instance: Optional[str] = None
    category: Optional[str] = None
    model: Optional[str] = None
    expected_file_hash: Optional[str] = None
    in_rooms: Optional[List[str]] = None
    target_name: Optional[str] = None
    target_bddl_instance: Optional[str] = None
    uuid_mode: str = "name_preserved"
    explicit_target_uuid: Optional[int] = None
    controller_joint_map: Optional[Dict[str, Any]] = None
    base_anchor_frame: Optional[Dict[str, Any]] = None
    required_for_restore: bool = True
    source_ref_index: Optional[int] = None

    def source_uuid(self) -> int:
        return compute_uuid(self.source_name)

    def record(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "bddl_instance": self.bddl_instance or "UNKNOWN",
            "category": self.category or "UNKNOWN",
            "model": self.model or "UNKNOWN",
            "expected_file_hash": self.expected_file_hash or "UNKNOWN",
            "in_rooms": self.in_rooms or "UNKNOWN",
            "target_name": self.target_name or "UNKNOWN",
            "target_bddl_instance": self.target_bddl_instance or "UNKNOWN",
            "uuid_mode": self.uuid_mode,
            "source_uuid": self.source_uuid(),
            "explicit_target_uuid": self.explicit_target_uuid
            if self.explicit_target_uuid is not None else "UNKNOWN",
            "controller_joint_map": _clean(self.controller_joint_map)
            if self.controller_joint_map else "UNKNOWN",
            "base_anchor_frame": _clean(self.base_anchor_frame)
            if self.base_anchor_frame else "UNKNOWN",
            "required_for_restore": self.required_for_restore,
        }


@dataclass
class RestorePlan:
    """Fail-closed result of the source-compat bootstrap check."""

    status: str
    mapped: List[Dict[str, Any]] = field(default_factory=list)
    blocked: List[Dict[str, Any]] = field(default_factory=list)
    pin_check: Dict[str, Any] = field(default_factory=dict)
    bootstrap_steps: Tuple[str, ...] = BOOTSTRAP_STEPS
    frame_checkpoints: Tuple[str, ...] = FRAME_CHECKPOINTS
    partial_success_allowed: bool = PARTIAL_SUCCESS_ALLOWED

    @property
    def ready(self) -> bool:
        return self.status == "ready" and not self.blocked


class SourceCompatReader:
    """CPU reader that builds the mapped restore plan from the frozen manifest.

    The live objects are duck-typed (same surface as `object_identity`); the
    reader never writes back to them or to the source hdf5 group.
    """

    def __init__(self, pin: SourcePin, entries: List[TargetMapEntry],
                 target_scene: Optional[Dict[str, Any]] = None):
        self.pin = pin
        self.entries = list(entries)
        self.target_scene = target_scene or {}
        self._source_group_was_written = False

    # -- identity refs -------------------------------------------------

    def _source_refs(self) -> List[SourceObjectRef]:
        refs: List[SourceObjectRef] = []
        for idx, entry in enumerate(self.entries):
            refs.append(SourceObjectRef(
                role=entry.bddl_instance or entry.source_name,
                source_name=entry.source_name,
                source_pose="UNKNOWN",
                bddl_instance=entry.bddl_instance,
                category=entry.category,
                model=entry.model,
                expected_file_hash=entry.expected_file_hash,
                in_rooms=entry.in_rooms,
                scene=SourceSceneRef(
                    scene_model=self.pin.scene_model,
                    scene_instance=self.pin.scene_instance,
                    config_attr_sha256=self.pin.config_attr_sha256,
                    scene_file_attr_sha256=self.pin.scene_file_attr_sha256,
                    versions=self.pin.versions,
                    episode=self.pin.episode,
                ),
                source_index=idx,
            ))
        return refs

    # -- uuid alignment ------------------------------------------------

    def _uuid_alignment(self, entry: TargetMapEntry,
                        live_name: Optional[str]) -> Dict[str, Any]:
        source_uuid = entry.source_uuid()
        if entry.uuid_mode == "explicit_remap":
            if entry.explicit_target_uuid is None:
                return {"aligned": False, "reason": REASON_REQUIRED_SEGMENT_UNMAPPED,
                        "source_uuid": source_uuid,
                        "detail": "explicit remap target uuid not evidenced"}
            target_uuid = int(entry.explicit_target_uuid)
            return {"aligned": True, "reason": "explicit_remap_evidenced",
                    "source_uuid": source_uuid, "target_uuid": target_uuid}

        # name_preserved: target name must equal source name; formula aligns.
        if live_name is None:
            return {"aligned": False, "reason": REASON_REQUIRED_SEGMENT_UNMAPPED,
                    "source_uuid": source_uuid, "detail": "no live target object"}
        if live_name != entry.source_name:
            return {"aligned": False, "reason": REASON_UUID_UNALIGNED_RENAMED,
                    "source_uuid": source_uuid, "live_name": live_name,
                    "detail": "name changed; explicit_remap required"}
        target_uuid = compute_uuid(live_name)
        if target_uuid != source_uuid:
            return {"aligned": False, "reason": REASON_FORMULA_ERROR,
                    "source_uuid": source_uuid, "target_uuid": target_uuid}
        return {"aligned": True, "reason": "name_preserved_uuid_formula_match",
                "source_uuid": source_uuid, "target_uuid": target_uuid}

    # -- public plan ---------------------------------------------------

    def prepare_restore(self, live_objects: List[Any],
                        scope_keys: Optional[List[Optional[str]]] = None,
                        live_scene_closure: Optional[Dict[str, Any]] = None,
                        source_attrs: Optional[Dict[str, Any]] = None) -> RestorePlan:
        pin_check = self.pin.matches(source_attrs)
        refs = self._source_refs()
        report = match_objects(
            refs, live_objects, scope_keys=scope_keys,
            live_scene_closure=live_scene_closure,
        )

        mapped: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []

        if pin_check["pin_match"] is False:
            blocked.append({
                "source_name": "source_pin",
                "reason": REASON_SOURCE_PIN_MISMATCH,
                "detail": pin_check,
            })

        for entry in self.entries:
            match = report["matches"].get(entry.source_name, {})
            verdict = match.get("verdict", "no_role_match")
            candidates = match.get("candidates", [])
            confirmed = next(
                (c for c in candidates
                 if c.get("verdict") in IDENTITY_CONFIRMED_VERDICTS),
                None)
            live_name = (confirmed or (candidates[0] if candidates else None))
            live_name = live_name.get("name") if live_name else match.get("matched_live_name")
            alignment = self._uuid_alignment(entry, live_name)
            record = entry.record()
            record.update({
                "verdict": verdict,
                "uuid_alignment": alignment,
                "candidate_count": match.get("candidate_count"),
            })

            reasons: List[str] = []
            if verdict not in IDENTITY_CONFIRMED_VERDICTS:
                if verdict in REJECTED_VERDICTS:
                    reasons.append(REASON_REJECTED)
                elif verdict in ("role_bound_scene_unverified",):
                    reasons.append(REASON_SCENE_UNVERIFIED)
                else:
                    reasons.append(REASON_ROLE_ONLY_NOT_IDENTITY)
            if not alignment["aligned"]:
                reasons.append(alignment["reason"])

            if reasons:
                record["reasons"] = reasons
                if entry.required_for_restore:
                    blocked.append(record)
                else:
                    mapped.append(record)
            else:
                mapped.append(record)

        status = "ready" if not blocked and pin_check["pin_match"] is not False \
            else "restore_blocked"
        return RestorePlan(
            status=status,
            mapped=mapped,
            blocked=blocked,
            pin_check=pin_check,
        )
