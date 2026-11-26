"""S2 decision contract: sanitizer, terminal predicate, argument validation,
and request-idempotency state machine.

Ported from the a28 frozen baseline ``s2_runtime.py`` stdlib-only contract.
The runtime deliberately does NOT import the simulator, write state, or launch
experiments. Network/subprocess/IPC concerns belong to the S2 harness (gap
re-bind), not this module.

Boundary rules preserved from the baseline:

- Strict JSON rejects duplicate keys and non-finite constants.
- FORBIDDEN observation fields are stripped before any model payload leaves
  the owner process.
- Exactly one whitelisted ``execute_primitive`` call; arguments stay inside
  the live schema and caps.
- A BDDL result is a dict; ``bool(dict)`` or Inside alone never means success.
- Stable request_id; at most two submit attempts; an unresolved same-id
  request blocks minting a new id on uncertainty.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

TERMINAL = {"completed", "rejected", "failed", "timed_out", "aborted_on_exception"}
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
ARGS = {"primitive", "target_object", "max_steps", "stop_condition", "hold_steps"}
FORBIDDEN = {
    "source_action_index", "source_index", "source_cursor", "source_cursor_next",
    "failure_action_index", "failure_index", "intervention_mode", "intervention_active",
    "intervention_start_source", "frozen_base_xy_offset_m", "base_xy_offset_m",
    "next_primitive", "next_primitive_suggestion", "source_action_indices",
    "failure_evidence", "source_reward", "recovery_reference_source_index",
    "execution_action_index", "failure_execution_index", "intervention",
}

MAX_SUBMIT_ATTEMPTS = 2


def strict_json(text: str) -> Any:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate_json_key_" + key)
            out[key] = value
        return out

    def constant(value):
        raise ValueError("nonfinite_json_" + value)

    return json.loads(text, object_pairs_hook=unique, parse_constant=constant)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                   allow_nan=False).encode()
    ).hexdigest()


def sanitized(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitized(v) for k, v in value.items() if k not in FORBIDDEN}
    if isinstance(value, list):
        return [sanitized(x) for x in value]
    return value


def capped_observation(obs: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    value = sanitized(obs)
    for name, meta in value.get("allowed_primitives_schema", {}).items():
        if name in cfg["primitive_caps"]:
            meta["max_steps"] = min(meta["max_steps"], cfg["primitive_caps"][name])
            if name == "continue_nominal_tracking" and value.get("active_skill_context", {}).get(
                "active_subtask"
            ) == "first_can_pick_and_place":
                meta["max_steps"] = min(meta["max_steps"], 3000)
            meta["default_max_steps"] = min(
                meta.get("default_max_steps", meta["max_steps"]), meta["max_steps"]
            )
    return value


def full_task_terminal(obs: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """Never bool(dict): BDDL success alone or all Inside alone is insufficient."""
    progress = obs.get("task_progress") or {}
    bddl = progress.get("bddl_done")
    if not isinstance(bddl, dict) or bddl.get("success") is not True:
        return False
    if progress.get("bucket_ever_held") is not True or progress.get("bucket_release_observed") is not True:
        return False
    for can in cfg["can_names"]:
        if any((progress.get(k) or {}).get(can) is not True for k in
               ("can_ever_held", "can_release_observed", "all_cans_inside")):
            return False
        if (obs.get("objects") or {}).get(can, {}).get("inside_bucket") is not True:
            return False
    arms = (obs.get("robot_state") or {}).get("arms") or {}
    for arm in ("left", "right"):
        state = arms.get(arm, {})
        if not {"object_in_hand", "constraint_present", "release_counter"}.issubset(state):
            return False
        if (state["object_in_hand"] is not None or state["constraint_present"] is not False
                or state["release_counter"] is not None):
            return False
    bucket = (obs.get("objects") or {}).get(cfg["bucket_name"], {})
    return bucket.get("on_ground") is True and any(
        "/floors_" in x for x in bucket.get("contact_bodies", [])
    )


def tool_schema(obs: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dynamic enums and limits; request identity is controlled by runtime, never model."""
    schema = obs.get("allowed_primitives_schema")
    if not isinstance(schema, dict) or not schema:
        raise ValueError("missing_live_primitive_schema")
    variants = []
    for primitive, meta in schema.items():
        if primitive not in cfg["allowed_primitives"]:
            continue
        maximum = min(meta["max_steps"], cfg["primitive_caps"][primitive])
        props: Dict[str, Any] = {
            "primitive": {"type": "string", "enum": [primitive]},
            "target_object": {"enum": meta["allowed_objects"]},
            "max_steps": {"type": "integer", "minimum": meta["min_steps"], "maximum": maximum},
        }
        if meta.get("allowed_stop_conditions") is not None:
            props["stop_condition"] = {"type": "string", "enum": meta["allowed_stop_conditions"]}
        if primitive == "stabilize_and_brake":
            props["hold_steps"] = {"type": "integer", "minimum": 1, "maximum": maximum}
        variants.append({
            "type": "object",
            "properties": props,
            "required": ["primitive", "target_object", "max_steps"]
            + (["stop_condition"] if primitive == "continue_nominal_tracking" else []),
            "additionalProperties": False,
        })
    return [{
        "type": "function",
        "function": {
            "name": "execute_primitive",
            "description": "Request exactly one bounded primitive; the S1 owner validates and executes it.",
            "parameters": {"type": "object", "oneOf": variants},
        },
    }]


def validate(args: Dict[str, Any], obs: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    obs = capped_observation(obs, cfg)
    if not isinstance(args, dict) or set(args) - ARGS:
        raise ValueError("unknown_tool_arguments")
    if not {"primitive", "target_object", "max_steps"}.issubset(args):
        raise ValueError("missing_tool_arguments")
    name = args["primitive"]
    if not isinstance(name, str):
        raise ValueError("primitive_type_invalid")
    meta = obs.get("allowed_primitives_schema", {}).get(name)
    if name not in cfg["allowed_primitives"] or not isinstance(meta, dict):
        raise ValueError("primitive_not_allowed")
    if args["target_object"] not in meta["allowed_objects"]:
        raise ValueError("object_not_allowed")
    steps = args["max_steps"]
    if type(steps) is not int or not meta["min_steps"] <= steps <= min(
        meta["max_steps"], cfg["primitive_caps"][name]
    ):
        raise ValueError("step_limit_invalid")
    if "stop_condition" in args and args["stop_condition"] not in (meta.get("allowed_stop_conditions") or []):
        raise ValueError("stop_condition_invalid")
    if "hold_steps" in args:
        if name != "stabilize_and_brake" or type(args["hold_steps"]) is not int \
                or not 1 <= args["hold_steps"] <= steps:
            raise ValueError("hold_steps_invalid")
    for arm in ("left", "right"):
        hand = obs.get("robot_state", {}).get("arms", {}).get(arm, {})
        if not {"object_in_hand", "constraint_present", "release_counter", "valid_sdk_hold"}.issubset(hand):
            raise ValueError("missing_actual_hand_state")
        if type(hand["constraint_present"]) is not bool or type(hand["valid_sdk_hold"]) is not bool:
            raise ValueError("invalid_actual_hand_flags")
    if name == "continue_nominal_tracking" and "stop_condition" not in args:
        raise ValueError("explicit_nominal_stop_required")
    if obs.get("runner_state") != "decision_required" or obs.get("decision_required") is not True \
            or obs.get("active_request") is not None:
        raise ValueError("not_an_idle_decision_boundary")


class RequestTracker:
    """Idempotency bookkeeping for S2 requests, independent of the IPC transport."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.pending: Optional[Dict[str, Any]] = None
        self.completed: List[Dict[str, Any]] = []

    def begin(self, request: Dict[str, Any]) -> str:
        rid = request.get("request_id")
        if not isinstance(rid, str) or not NAME.match(rid):
            raise ValueError("invalid_persisted_request_id")
        if request.get("run_id") != self.run_id:
            raise ValueError("request_run_id_mismatch")
        for done in self.completed:
            if done["request"]["request_id"] == rid:
                return "replayed_completed"
        if self.pending is not None and self.pending["request"]["request_id"] == rid:
            raise ValueError("unresolved_same_id_blocked")
        attempts = self.pending.get("submit_attempts", 0) if self.pending else 0
        if self.pending is not None and self.pending["request"]["request_id"] != rid:
            raise ValueError("active_request_blocks_new_id")
        if attempts >= MAX_SUBMIT_ATTEMPTS:
            raise ValueError("submit_attempts_exhausted")
        self.pending = {"request": request, "submit_attempts": attempts + 1}
        return "accepted_pending"

    def receipt(self, response: Dict[str, Any]) -> None:
        if self.pending is None:
            raise ValueError("receipt_without_pending_request")
        if response.get("request_id") != self.pending["request"]["request_id"] \
                or response.get("run_id") != self.run_id:
            raise ValueError("receipt_identity_mismatch")
        self.completed.append({"request": self.pending["request"], "response": response})
        self.pending = None

    def mark_unknown(self) -> Dict[str, Any]:
        """Acceptance unknown: keep pending, do not mint a new request id."""
        if self.pending is None:
            raise ValueError("mark_unknown_without_pending")
        self.pending["acceptance"] = "unknown"
        return {"status": "acceptance_unknown", "request_id": self.pending["request"]["request_id"]}

    def retry_allowed(self) -> bool:
        return self.pending is not None and \
            self.pending.get("submit_attempts", 0) < MAX_SUBMIT_ATTEMPTS
