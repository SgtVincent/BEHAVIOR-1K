"""Saved delta-TF policy using upstream policy / evaluator base classes.

The official evaluator owns env.step, metrics and video. This module does not
implement a parallel evaluation loop or a websocket / model-serving stack.

Two read-only saved-row sources are accepted:

- ``converted_dir``: the original converted episode directory holding
  ``manifest.json`` and ``actions.npy`` (schema
  ``brvla_saved_execution23_v1``). The manifest gates schema, representation,
  shape, dtype and action digest; this lane reads the original npy, not a
  rebuilt archive.
- ``npz_path`` / ``array_key``: an explicit compatibility archive (e.g. a
  site-side npz wrap that preserves the original npy sha). Only width and
  finiteness are checked there; the manifest contract gates belong to the
  converted-dir lane.

Model chunk rows (``chunk_cumulative``, physical23) are a separate frozen-model
lane; they are never read through this replay policy.
"""
import hashlib
import json
import os
import pathlib
import traceback

import numpy as np
import torch as th
from hydra.utils import instantiate

from omnigibson.action_primitives.delta_tf_action_primitives import (
    GRIPPER_RANGE_M,
    DeltaTFActionPrimitives,
)
from omnigibson.eval.evaluator import BatchedEvaluator
from omnigibson.eval.policies import LocalPolicy


SAVED_SCHEMA = "brvla_saved_execution23_v1"
SAVED_REPRESENTATION = "pairwise_delta"
ACTION23 = 23
# Control loop runs at the sim-step frequency; physics substeps run faster.
# This slice observes control rounds and policy actions only; it does not count
# cumulative physics substeps, so physical_steps is reported null rather than
# inferred from the control counter.
CONTROL_HZ = 30
PHYSICS_HZ = 120


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_run_failure(evaluator, json_dir, task_name, rollout_id, exc):
    """Persist a pre-completion failure without swallowing it.

    Writes the policy bound receipt and a failure JSON, and flushes/closes any
    already-open official video writer through the evaluator's official setter.
    The caller re-raises the exception: no action is synthesized, no step is
    invented, and the failure is not reported as success.
    """
    flushed = False
    for state in getattr(evaluator, "instance_eval_states", []):
        if getattr(state, "video_writer", None) is not None:
            evaluator._set_video_writer(state, None)
            flushed = True
    policy = getattr(evaluator, "policy", None)
    receipt = None
    if policy is not None and hasattr(policy, "bound_receipt"):
        receipt = policy.bound_receipt()
        receipt["exhaustion_notice"] = getattr(evaluator, "_exhaustion_notice", None)
    control_steps = getattr(evaluator, "_control_steps", 0)
    trials_recorded = getattr(evaluator, "n_trials", 0)
    plans_attempted = receipt.get("plans_attempted", 0) if receipt else 0
    plans_succeeded = receipt.get("plans_succeeded", 0) if receipt else 0
    commands_issued = receipt.get("commands_issued", 0) if receipt else 0
    failure = {
        "task": task_name,
        "rollout_id": rollout_id,
        "failed": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "trials_recorded": trials_recorded,
        # One control step = one successful evaluator `_step_fn` round, i.e. one
        # `env.step` call with one policy action. Physics substeps run at
        # PHYSICS_HZ inside that call and are not counted here.
        "control_steps": control_steps,
        "physical_steps": None,
        "control_hz": CONTROL_HZ,
        "physics_hz": PHYSICS_HZ,
        "plans_attempted": plans_attempted,
        "plans_succeeded": plans_succeeded,
        "commands_issued": commands_issued,
        # Booleans are derived from the real counters above, not assumed False.
        "plan_succeeded": plans_succeeded > 0,
        "command_issued": commands_issued > 0,
        "env_step_applied": control_steps > 0,
        "video_flushed": flushed,
    }
    if receipt is not None:
        receipt["failure"] = failure
    if json_dir is not None:
        failure_path = os.path.join(json_dir, f"{task_name}_failure_{rollout_id}.json")
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2, default=float)
        if receipt is not None:
            receipt_path = os.path.join(json_dir, f"{task_name}_receipt_{rollout_id}.json")
            with open(receipt_path, "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2, default=float)
    return failure


class DeltaTFReplayPolicy(LocalPolicy):
    """Tracking-only, fixed-base replay adapter; not full BRVLA admission.

    Initial-state restore and producer/checkpoint lineage belong to the caller.
    Finite exhaustion raises StopIteration rather than repeating source rows or
    returning zero joint targets. The evaluator must mark it as a diagnostic
    exit, not a task-success result.
    """

    def __init__(
        self,
        env,
        robot,
        tool_links,
        representation,
        npz_path=None,
        array_key=None,
        converted_dir=None,
        manifest_sha256=None,
        actions_sha256=None,
        **kwargs,
    ):
        super().__init__(action_dim=robot.action_dim)
        if converted_dir:
            self.rows, self._source_receipt = self._load_converted(
                converted_dir,
                manifest_sha256=manifest_sha256,
                actions_sha256=actions_sha256,
                representation=representation,
            )
        elif npz_path:
            self.rows, self._source_receipt = self._load_npz(npz_path, array_key)
        else:
            raise ValueError("converted_dir or npz_path/array_key must be provided")

        self.primitives = DeltaTFActionPrimitives(
            env, robot, tool_links, representation=representation
        )
        self.reset()

    # -- saved-row loading and contract gates ---------------------------------

    def _load_converted(self, converted_dir, *, manifest_sha256, actions_sha256, representation):
        root = pathlib.Path(converted_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"converted manifest absent: {manifest_path}")

        actual_manifest_sha = _sha256_file(manifest_path)
        if manifest_sha256 is not None and actual_manifest_sha != manifest_sha256:
            raise ValueError(
                f"manifest sha drift: expected {manifest_sha256}, got {actual_manifest_sha}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("schema") != SAVED_SCHEMA:
            raise ValueError(f"unsupported saved schema {manifest.get('schema')!r}")
        manifest_repr = manifest.get("representation")
        if manifest_repr != SAVED_REPRESENTATION:
            raise ValueError(
                f"replay lane only accepts {SAVED_REPRESENTATION!r}; "
                f"manifest carries {manifest_repr!r}"
            )
        if representation != manifest_repr:
            raise ValueError(
                f"requested representation {representation!r} != manifest {manifest_repr!r}"
            )
        shape = manifest.get("shape")
        if not shape or int(shape[1]) != ACTION23:
            raise ValueError(f"manifest shape {shape} != (N, {ACTION23})")
        if manifest.get("dtype") not in ("<f4", "f4", "float32"):
            raise ValueError(f"manifest dtype {manifest.get('dtype')!r} != float32 (<f4)")

        action_file = manifest.get("action_file", "actions.npy")
        actions_path = root / action_file
        if not actions_path.exists():
            raise ValueError(f"actions file absent: {actions_path}")
        actual_actions_sha = _sha256_file(actions_path)
        manifest_actions_sha = manifest.get("action_sha256")
        if manifest_actions_sha is not None and actual_actions_sha != manifest_actions_sha:
            raise ValueError(
                f"actions sha drift vs manifest: {actual_actions_sha} != {manifest_actions_sha}"
            )
        if actions_sha256 is not None and actual_actions_sha != actions_sha256:
            raise ValueError(
                f"actions sha drift: expected {actions_sha256}, got {actual_actions_sha}"
            )

        array = np.load(actions_path)
        if array.ndim != 2 or array.shape[1] != ACTION23 or len(array) == 0:
            raise ValueError(f"converted data shape {tuple(array.shape)} != nonempty (N, 23)")
        if array.dtype != np.float32:
            raise ValueError(f"converted data dtype {array.dtype} != float32")
        if not np.isfinite(array).all():
            raise ValueError("Converted data contains non-finite entries")

        chunks = manifest.get("chunks") or []
        anchors = chunks[0].get("anchors") if chunks else None
        provenance = manifest.get("provenance") or {}
        receipt = {
            "source": "converted_dir",
            "schema": manifest.get("schema"),
            "representation": manifest_repr,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "dt_seconds": manifest.get("dt_seconds"),
            "columns": manifest.get("columns"),
            "anchors": anchors,
            "chest_link": provenance.get("chest_link"),
            "manifest_sha256": actual_manifest_sha,
            "actions_sha256": actual_actions_sha,
        }
        return np.array(array, copy=True), receipt

    def _load_npz(self, npz_path, array_key):
        if not array_key:
            raise ValueError("array_key is required with npz_path")
        with np.load(npz_path, allow_pickle=False) as data:
            rows = np.array(data[array_key], copy=True)
        if rows.ndim != 2 or rows.shape[1] != ACTION23 or len(rows) == 0:
            raise ValueError("Converted data must be a nonempty Nx23 array")
        if not np.isfinite(rows).all():
            raise ValueError("Converted data contains non-finite entries")
        receipt = {
            "source": "npz_compat",
            "npz_path": npz_path,
            "array_key": array_key,
            "shape": list(rows.shape),
        }
        return rows, receipt

    # -- evaluator policy interface --------------------------------------------

    def reset(self):
        self._index = 0
        self._generator = None
        self._anchored = False
        self._commands_issued = 0

    def forward(self, obs, *args, **kwargs):
        # BatchedEvaluator(1) still expects a leading batch dimension.
        if not self._anchored:
            self.primitives.bind_anchors()
            self._anchored = True
        while True:
            if self._generator is None:
                if self._index >= len(self.rows):
                    raise StopIteration("Delta-TF input exhausted; task success is not inferred")
                self._generator = self.primitives.apply(self.rows[self._index])
            try:
                action = next(self._generator)
            except StopIteration:
                self._index += 1
                self._generator = None
                continue
            action = th.as_tensor(action, dtype=th.float32).detach().cpu()
            if action.shape != (self.action_dim,) or not th.isfinite(action).all():
                raise ValueError("Upstream primitive yielded an invalid controller command")
            self._commands_issued += 1
            return action.unsqueeze(0)

    def bound_receipt(self):
        receipt = {"index": self._index, "rows": len(self.rows), **self._source_receipt}
        receipt["commands_issued"] = self._commands_issued
        profile = getattr(self.primitives, "_curobo_profile_receipt", None)
        if profile is not None:
            receipt["curobo_profile"] = profile
        receipt["plans_attempted"] = getattr(self.primitives, "_plans_attempted", 0)
        receipt["plans_succeeded"] = getattr(self.primitives, "_plans_succeeded", 0)
        return receipt


class LiveTargetDiagnosticPolicy(LocalPolicy):
    """First-plan smoke diagnostic on the live robot; NOT a replay / frame0 /
    scientific pass.

    One synthetic execution23 row is built from live link poses: small nonzero
    local translations for left/right/chest, zero base displacement, and grip
    widths held at the live finger qpos mean. It still flows through the
    official primitives, cuRobo MotionGen and controller execution; the
    evaluator owns stepping, metrics and video. No original episode data,
    source anchors, or frame0 state are read or restored here.
    """

    DEFAULT_OFFSETS = {
        "left": (0.02, 0.0, 0.0),
        "right": (0.0, 0.02, 0.0),
        "chest": (0.0, 0.0, 0.01),
    }

    def __init__(self, env, robot, tool_links, representation="pairwise_delta", offsets=None, **kwargs):
        super().__init__(action_dim=robot.action_dim)
        self.primitives = DeltaTFActionPrimitives(
            env, robot, tool_links, representation=representation
        )
        self._offsets = dict(self.DEFAULT_OFFSETS)
        if offsets:
            self._offsets.update(offsets)
        self._source_receipt = {
            "source": "live_diagnostic",
            "scientific_pass": False,
            "replay_pass": False,
            "frame0_pass": False,
            "grip_mode": "hold_current_finger_qpos",
            "grip_physical_range_unverified": True,
            "grip_metre_width_claim": False,
            "note": (
                "Synthetic live small-offset target; no source episode, anchor or frame0 restore. "
                "Grip joints are held at their current finger qpos through the official plan; "
                "no new grip target is issued. Finger qpos is NOT a verified physical metre width."
            ),
        }
        self.reset()

    def reset(self):
        self._index = 0
        self._generator = None
        self._anchored = False
        self._row = None
        self._commands_issued = 0

    def _build_diagnostic_row(self):
        robot = self.primitives.robot
        # Hold the current finger qpos as the grip reference. These slot values
        # populate the execution23 grip channels so the unchanged-grip guard
        # passes; they are a qpos hold reference, NOT a physical metre-width
        # command, and the metre range is unverified (see receipt flags).
        widths = []
        for arm in ("left", "right"):
            measured = robot.get_joint_positions()[robot.gripper_control_idx[arm]]
            widths.append(float(th.clamp(measured.mean(), 0.0, GRIPPER_RANGE_M).item()))
        row = th.zeros(ACTION23, dtype=th.float32)
        for name, sl in (("left", slice(0, 6)), ("right", slice(6, 12)), ("chest", slice(17, 23))):
            dx, dy, dz = self._offsets[name]
            row[sl] = th.tensor([dx, dy, dz, 0.0, 0.0, 0.0], dtype=th.float32)
        row[12:14] = th.tensor(widths, dtype=th.float32)
        row[14:17] = th.zeros(3, dtype=th.float32)
        if not th.isfinite(row).all():
            raise ValueError("Diagnostic row is non-finite")
        return row

    def forward(self, obs, *args, **kwargs):
        if not self._anchored:
            self.primitives.bind_anchors()
            self._anchored = True
        while True:
            if self._generator is None:
                if self._index >= 1:
                    raise StopIteration("Live diagnostic target exhausted; this is not task success")
                self._row = self._build_diagnostic_row()
                self._generator = self.primitives.apply(self._row)
            try:
                action = next(self._generator)
            except StopIteration:
                self._index += 1
                self._generator = None
                continue
            action = th.as_tensor(action, dtype=th.float32).detach().cpu()
            if action.shape != (self.action_dim,) or not th.isfinite(action).all():
                raise ValueError("Upstream primitive yielded an invalid controller command")
            self._commands_issued += 1
            return action.unsqueeze(0)

    def bound_receipt(self):
        receipt = {"index": self._index, "rows": 1, **self._source_receipt}
        receipt["commands_issued"] = self._commands_issued
        profile = getattr(self.primitives, "_curobo_profile_receipt", None)
        if profile is not None:
            receipt["curobo_profile"] = profile
        receipt["plans_attempted"] = getattr(self.primitives, "_plans_attempted", 0)
        receipt["plans_succeeded"] = getattr(self.primitives, "_plans_succeeded", 0)
        receipt["offsets"] = {k: list(v) for k, v in self._offsets.items()}
        return receipt


class PrimitiveEvaluator(BatchedEvaluator):
    """Only override the policy factory to pass the existing live env / robot.

    Upstream load/reset/metrics/run/video handling remains unchanged. This is an
    engineering replay evaluator; policies using ground-truth scene geometry are
    not automatically challenge-observation-compliant model evaluations.
    """

    def load_policy(self):
        if self.num_envs != 1:
            raise ValueError("This baseline slice requires one logical environment")
        robot = self.instance_eval_states[0].env_accessor.robot
        return instantiate(self.cfg.model, env=self.env, robot=robot)

    def _step_fn(self, active_env_indices):
        """Catch policy input exhaustion so the official record_fn still runs:
        the active env is marked truncated, the result JSON is written and the
        video writer flushed by the upstream loop. No further actions are
        synthesized; already taken steps remain the only evidence.
        """
        try:
            terminated, truncated = super()._step_fn(active_env_indices)
        except StopIteration as exc:
            terminated = th.zeros(self.num_envs, dtype=th.bool)
            truncated = th.zeros(self.num_envs, dtype=th.bool)
            for idx in active_env_indices:
                truncated[idx] = True
            self._exhaustion_notice = str(exc)
            return terminated, truncated
        self._control_steps = getattr(self, "_control_steps", 0) + 1
        return terminated, truncated

