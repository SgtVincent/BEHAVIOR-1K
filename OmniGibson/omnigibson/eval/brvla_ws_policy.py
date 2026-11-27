"""Peer websocket policy bound to the original frozen BRVLA model server.

Scope (segment 2 of the baseline migration):

- The original checkpoint, stats, preprocessing, chunk/anchor/history and
  control clock remain in the unchanged frozen server process. This client
  side never loads weights, opens stats, or rebuilds the model serving stack.
- Transport reuses the upstream :class:`WebsocketClientPolicy` (healthz poll,
  ws handshake, msgpack ndarray envelope, optional open-loop chunk replay).
- This peer only (a) verifies the frozen server metadata gate at reset,
  (b) maps the batched official evaluator observation onto the frozen wire
  keys, (c) validates the returned 23-wide finite row, and (d) holds the
  physical23 / IK-target23 row at the controller boundary until a bound
  native-joint decoder seam exists.

Fail-closed: a missing key, width mismatch, non-finite value, unverified
metadata or absent decoder seam raises before any simulator step. No source
joint-angle fallback, no state padding, no guessed image packing.

cuRobo v2 / MotionPlanner is not involved in this path.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from omnigibson.eval.policies import WebsocketPolicy


# ---- frozen wire contract (read from the retained original bridge sources) --

ACTION23 = 23
LEGACY_PROPRIO_WIDTH = 256
POST1_PROPRIO_WIDTH = 61

# Schema labels share the 23 width but not semantics: the frozen server row is
# an IK-target23 physical plan; only a verified joint-native23 command may reach
# the controller boundary.
JOINT_NATIVE23 = "joint-native23"
IK_TARGET23 = "IK-target23"

# Official post1 61-d proprio order, verified against
# ``omnigibson.eval.utils.eval_utils.PROPRIOCEPTION_INDICES["R1Pro"]``.
POST1_SL = {
    "base_qvel": slice(0, 3),
    "arm_left_qpos": slice(3, 10),
    "gripper_left_qpos": slice(24, 26),
    "arm_right_qpos": slice(28, 35),
    "gripper_right_qpos": slice(49, 51),
    "trunk_qpos": slice(53, 57),
}

# Legacy 256-d wire order consumed by the frozen online adapter. EE slices are
# profile-zero under the frozen ``remove_eef_state=true`` topology; they enter
# the wire but are not numerically consumed on that profile.
LEGACY_SL = {
    "robot_lin_vel": slice(152, 155),
    "robot_ang_vel": slice(155, 158),
    "arm_left_qpos": slice(158, 165),
    "eef_left_pos": slice(186, 189),
    "eef_left_quat": slice(189, 193),
    "gripper_left_qpos": slice(193, 195),
    "arm_right_qpos": slice(197, 204),
    "eef_right_pos": slice(225, 228),
    "eef_right_quat": slice(228, 232),
    "gripper_right_qpos": slice(232, 234),
    "trunk_qpos": slice(236, 240),
}

# (post1 key, legacy key, role). ``bound`` slices are copied from the official
# proprio row; ``profile_zero`` slots stay declared zeros on the frozen profile.
_BINDINGS: List[Tuple[Optional[str], str, str]] = [
    ("arm_left_qpos", "arm_left_qpos", "bound"),
    ("gripper_left_qpos", "gripper_left_qpos", "bound"),
    ("arm_right_qpos", "arm_right_qpos", "bound"),
    ("gripper_right_qpos", "gripper_right_qpos", "bound"),
    ("trunk_qpos", "trunk_qpos", "bound"),
    (None, "eef_left_pos", "profile_zero_eef"),
    (None, "eef_left_quat", "profile_zero_eef"),
    (None, "eef_right_pos", "profile_zero_eef"),
    (None, "eef_right_quat", "profile_zero_eef"),
]

# Frozen server metadata, checked at reset against the server's first frame.
REQUIRED_METADATA: Dict[str, Any] = {
    "action_layout": "physical23_same_anchor_cumulative",
    "horizon": 32,
    "chunk_generated": 50,
    "history_every_control_step": True,
}

# Default r1pro sensor role -> sensor prim name (mirrors eval/r1pro.yaml).
DEFAULT_CAMERAS: Dict[str, str] = {
    "head": "zed_link:Camera:0",
    "left_wrist": "left_realsense_link:Camera:0",
    "right_wrist": "right_realsense_link:Camera:0",
}


class CheckpointWsContractError(RuntimeError):
    """Fail-closed bridge contract violation; never silently zero-filled."""


class CheckpointActionBlocked(CheckpointWsContractError):
    """A physical23 / IK-target23 row reached the step boundary without a legal
    native-joint decoder seam. The held row takes zero simulator steps."""


def _to_plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _finite(values: List[float]) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def _sha_bytes(values: Any) -> str:
    blob = json.dumps(values, sort_keys=True, allow_nan=False).encode("utf-8")
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def _squeeze_single_batch(value: Any) -> Any:
    """Drop the leading num_envs axis for the gated single-instance batch."""
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1 and shape[0] == 1:
        return value[0]
    plain = _to_plain(value)
    if isinstance(plain, list) and len(plain) == 1:
        return plain[0]
    return value


class CheckpointWsPolicy(WebsocketPolicy):
    """Evaluator-facing peer around the upstream websocket client.

    Constructor kwargs mirror ``omnigibson.eval.policies.WebsocketPolicy`` so the
    official hydra ``instantiate(self.cfg.model)`` seam can build it directly.

    Args:
        host / port / allow_reconnect / action_chunk_size: forwarded to the
            upstream client. ``action_chunk_size <= 1`` keeps one request per
            control step.
        robot_name: evaluator robot name prefix for flattened obs keys.
        cameras: role -> sensor prim name mapping. Defaults to the r1pro profile.
        decoder: optional duck-typed seam exposing
            ``decode(tagged_action, wire=...) -> native joint command``. Absent
            by default: the physical23 row is held and zero env steps are taken.
        trace_path: optional JSONL trace path; falls back to ``BR_WS_TRACE``.
    """

    def __init__(
        self,
        *args: Any,
        host: Optional[str] = None,
        port: Optional[int] = None,
        allow_reconnect: bool = False,
        action_chunk_size: int = 0,
        robot_name: str = "robot_r1",
        cameras: Optional[Dict[str, str]] = None,
        decoder: Any = None,
        trace_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            host=host,
            port=port,
            allow_reconnect=allow_reconnect,
            action_chunk_size=action_chunk_size,
            **kwargs,
        )
        self._robot_name = robot_name
        self._cameras = dict(DEFAULT_CAMERAS if cameras is None else cameras)
        self._decoder = decoder
        self._trace_path = trace_path or os.environ.get("BR_WS_TRACE")
        self._step = 0
        self._reset_epoch = 0
        self._metadata_verified = False
        self._eval_action_dim: Optional[int] = None
        self._prop_key = f"{self._robot_name}::proprio"
        self._cam_keys = {
            role: f"{self._robot_name}::{sensor}::rgb"
            for role, sensor in self._cameras.items()
        }
        self._emit({
            "event": "ws_policy_constructed",
            "robot_name": self._robot_name,
            "camera_keys": list(self._cam_keys.values()),
            "decoder_bound": self._decoder is not None,
            "action_chunk_size": int(action_chunk_size),
        })

    # -- evaluator policy interface ------------------------------------------

    def set_action_dim(self, action_dim: Optional[int]) -> None:
        if action_dim is not None and int(action_dim) != ACTION23:
            raise CheckpointWsContractError(
                f"evaluator action_dim={action_dim} != frozen {ACTION23}; "
                "refuse to construct a mismatched policy"
            )
        self._eval_action_dim = None if action_dim is None else int(action_dim)

    def reset(self) -> None:
        super().reset()
        self._step = 0
        self._reset_epoch += 1
        self._metadata_verified = False
        self._verify_metadata()
        if hasattr(self._decoder, "reset"):
            self._decoder.reset(epoch=f"reset#{self._reset_epoch}")
        self._emit({
            "event": "ws_policy_reset",
            "env_step": 0,
            "reset_epoch": f"reset#{self._reset_epoch}",
            "metadata_verified": self._metadata_verified,
        })

    def forward(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if self.policy is None:
            raise CheckpointWsContractError("no websocket client bound; host/port are required")
        if not self._metadata_verified:
            raise CheckpointWsContractError("server interface unverified; refuse to send a query")

        need_new = self._client_chunk_exhausted()
        wire, lineage = self._build_wire(obs, self._step, need_new)

        action = self.policy.act(wire)
        row, response_shape = self._validate_response(action)

        provenance = {
            "env_step": self._step,
            "reset_epoch": f"reset#{self._reset_epoch}",
            "row_sha256": _sha_bytes(row),
            "layout": REQUIRED_METADATA["action_layout"],
        }
        tagged = {"action": row, "provenance": provenance, "wire_keys": sorted(wire.keys())}

        if self._decoder is None:
            self._emit({
                "event": "action_blocked",
                "env_step": self._step,
                "reason": "decoder_seam_absent",
                "query_success": True,
                "held": True,
                "env_steps": 0,
                "action_len": len(row),
                "schema_label": "IK-target23",
                "layout": REQUIRED_METADATA["action_layout"],
            })
            raise CheckpointActionBlocked(
                "no legal native-joint decoder seam; physical23 row held, zero env steps")

        native = self._decode(tagged, wire)
        native_vector = self._validate_native(native)

        provenance = getattr(native, "provenance", None) or provenance
        self._emit({
            "event": "ws_policy_forward",
            "env_step": self._step,
            "need_new_action": bool(need_new),
            "wire_keys": lineage["wire_keys"],
            "response": {
                "action_len": len(native_vector),
                "shape": response_shape,
                "finite": True,
                "action_row_sha256": _sha_bytes(native_vector),
                "provenance": provenance,
            },
        })
        self._step += 1

        import torch as th

        return th.as_tensor(native_vector, dtype=th.float32).detach().cpu().unsqueeze(0)

    act = forward

    # -- wire assembly ---------------------------------------------------------

    def _build_wire(
        self, obs: Dict[str, Any], env_step: int, need_new_action: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._prop_key not in obs:
            raise CheckpointWsContractError(f"missing frozen proprio key {self._prop_key}")
        legacy, proprio_lineage = self._map_proprio(obs[self._prop_key])

        image_lineage: List[Dict[str, Any]] = []
        images: Dict[str, Any] = {}
        for role, cam_key in self._cam_keys.items():
            if cam_key not in obs:
                raise CheckpointWsContractError(f"missing frozen camera key {cam_key}")
            arr, shape, digest = self._image_to_wire(obs[cam_key])
            images[cam_key] = arr
            image_lineage.append({
                "role": role,
                "key": cam_key,
                "shape": shape,
                "channels": shape[-1],
                "sha256": digest,
            })

        wire: Dict[str, Any] = {
            self._prop_key: legacy,
            **images,
            "env_step": int(env_step),
            "need_new_action": bool(need_new_action),
        }
        lineage = {
            "proprio": proprio_lineage,
            "images": image_lineage,
            "wire_keys": sorted(wire.keys()),
        }
        return wire, lineage

    def _map_proprio(self, row: Any) -> Tuple[List[float], Dict[str, Any]]:
        plain = _to_plain(_squeeze_single_batch(row))
        if not isinstance(plain, list) or len(plain) != POST1_PROPRIO_WIDTH:
            raise CheckpointWsContractError(
                f"post1 proprio width {len(plain) if isinstance(plain, list) else 'n/a'} "
                f"!= {POST1_PROPRIO_WIDTH}; refuse to reshape or zero-pad")
        if not _finite(plain):
            raise CheckpointWsContractError("post1 proprio contains non-finite values")

        legacy = [0.0] * LEGACY_PROPRIO_WIDTH
        lineage: Dict[str, Any] = {"bindings": [], "profile_zero": []}

        # Base velocity: official base_qvel is already robot-local [vx, vy, wz].
        bv = [float(v) for v in plain[POST1_SL["base_qvel"]]]
        legacy[LEGACY_SL["robot_lin_vel"]] = [bv[0], bv[1], 0.0]
        legacy[LEGACY_SL["robot_ang_vel"]] = [0.0, 0.0, bv[2]]
        lineage["bindings"].append({
            "legacy_key": "robot_lin_vel", "legacy_slice": [152, 155],
            "post1_key": "base_qvel", "post1_slice": [0, 2],
            "role": "bound", "frame": "robot_local_base",
            "source_sha256": _sha_bytes(bv[0:2]),
        })
        lineage["bindings"].append({
            "legacy_key": "robot_ang_vel", "legacy_slice": [155, 158],
            "post1_key": "base_qvel", "post1_slice": [2, 3],
            "role": "bound", "frame": "robot_local_base",
            "source_sha256": _sha_bytes(bv[2:3]),
        })
        # Planar-holonomic-base profile zeros: no vertical / roll / pitch DOF.
        lineage["profile_zero"].append({
            "legacy_key": "robot_lin_vel_z", "legacy_slice": [154, 155],
            "role": "profile_zero_planar_base",
        })
        lineage["profile_zero"].append({
            "legacy_key": "robot_ang_vel_roll_pitch", "legacy_slice": [155, 157],
            "role": "profile_zero_planar_base",
        })

        for post1_key, legacy_key, role in _BINDINGS:
            sl = LEGACY_SL[legacy_key]
            if role == "bound":
                src = [float(v) for v in plain[POST1_SL[post1_key]]]
                legacy[sl] = src
                lineage["bindings"].append({
                    "legacy_key": legacy_key,
                    "legacy_slice": [sl.start, sl.stop],
                    "post1_key": post1_key,
                    "post1_slice": [POST1_SL[post1_key].start, POST1_SL[post1_key].stop],
                    "role": role,
                    "source_sha256": _sha_bytes(src),
                })
            else:
                legacy[sl] = [0.0] * (sl.stop - sl.start)
                lineage["profile_zero"].append({
                    "legacy_key": legacy_key,
                    "legacy_slice": [sl.start, sl.stop],
                    "role": role,
                })

        lineage["legacy_width"] = LEGACY_PROPRIO_WIDTH
        lineage["legacy_nonzero"] = sum(1 for v in legacy if float(v) != 0.0)
        lineage["legacy_sha256"] = _sha_bytes(legacy)
        return legacy, lineage

    def _image_to_wire(self, value: Any) -> Tuple[Any, List[int], str]:
        arr = value
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu()
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
        elif hasattr(arr, "shape") and not isinstance(arr, (list, tuple)):
            import numpy as np

            arr = np.asarray(arr)
        arr = _squeeze_single_batch(arr)

        plain = _to_plain(arr)
        shape = list(getattr(arr, "shape", None) or _shape_plain(plain))
        if not shape or shape[-1] not in (3, 4):
            raise CheckpointWsContractError(
                f"camera frame channel depth {shape[-1:] if shape else 'n/a'} != 3/4; "
                "refuse to guess image packing")
        if shape[-1] == 4:
            if isinstance(arr, list):
                plain = _strip_alpha(plain)
                shape[-1] = 3
            else:
                import numpy as np

                arr = np.asarray(arr)[..., :3]
                plain = _to_plain(arr)
                shape = list(arr.shape)
        digest = _sha_bytes(plain)
        return plain, shape, digest

    # -- response / boundary validation ----------------------------------------

    def _validate_response(self, action: Any) -> Tuple[List[float], List[int]]:
        plain = _to_plain(_squeeze_single_batch(action))
        if isinstance(plain, list) and plain and isinstance(plain[0], list):
            if len(plain) != 1:
                raise CheckpointWsContractError(
                    f"server response batch dim {len(plain)} != 1; refuse")
            row = plain[0]
            shape = [len(plain), len(row)]
        else:
            row = plain
            shape = [len(row)]
        if len(row) != ACTION23 or not _finite(row):
            raise CheckpointWsContractError(
                f"server response action invalid: len={len(row)} finite={_finite(row)}")
        return [float(v) for v in row], shape

    def _validate_native(self, native: Any) -> List[float]:
        label = getattr(native, "label", None)
        if label is not None and label != JOINT_NATIVE23:
            raise CheckpointActionBlocked(
                f"decoder handed back {label!r} at the step boundary; "
                "only a verified joint-native23 command may reach env.step"
            )
        vector = getattr(native, "vector", None)
        plain = _to_plain(vector if vector is not None else native)
        if isinstance(plain, list) and plain and isinstance(plain[0], list):
            plain = plain[0]
        if not isinstance(plain, list) or len(plain) != ACTION23 or not _finite(plain):
            raise CheckpointWsContractError(
                "decoder returned an invalid native23 command; refuse to step")
        return [float(v) for v in plain]

    def _decode(self, tagged: Dict[str, Any], wire: Dict[str, Any]) -> Any:
        native = self._decoder.decode(tagged, wire=wire)
        if native is None:
            raise CheckpointActionBlocked("decoder returned no native23 command")
        return native

    # -- internals -------------------------------------------------------------

    def _verify_metadata(self) -> None:
        meta = self.policy.get_server_metadata() or {}
        if not meta:
            raise CheckpointWsContractError("server metadata empty; interface unverified")
        for key, expected in REQUIRED_METADATA.items():
            if key not in meta:
                raise CheckpointWsContractError(f"server metadata missing key {key!r}")
            if meta.get(key) != expected:
                raise CheckpointWsContractError(
                    f"server metadata {key} mismatch: expected={expected!r} "
                    f"actual={meta.get(key)!r}")
        self._metadata_verified = True

    def _client_chunk_exhausted(self) -> bool:
        """Replan boundary from the upstream client's own chunk queue, if exposed.

        No independent mirror queue mints a row index; absent client internals
        default to a fresh query every control step.
        """
        chunk = getattr(self.policy, "_action_chunk", None)
        index = getattr(self.policy, "_action_chunk_index", None)
        if chunk is None or not isinstance(index, int):
            return True
        length = getattr(chunk, "shape", None)
        if length is not None and len(length) >= 2:
            return index >= int(length[-2])
        return True

    def _emit(self, event: Dict[str, Any]) -> None:
        if not self._trace_path:
            return
        import time

        event["at_unix"] = time.time()
        path = pathlib.Path(self._trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")


def _strip_alpha(frame: Any) -> Any:
    if isinstance(frame, list) and frame and isinstance(frame[0], list):
        return [_strip_alpha(v) for v in frame]
    if isinstance(frame, list):
        return list(frame)[:3]
    return frame


def _shape_plain(plain: Any) -> List[int]:
    shape: List[int] = []
    cur = plain
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return shape
