"""Official-evaluator policy backed by the managed replay wrapper.

``ManagedReplayPolicy`` is instantiated by the post1 ``BatchedEvaluator``
through the same hydra ``_target_`` seam as ``LocalPolicy`` /
``WebsocketPolicy``. The evaluator keeps its own reset / observation / step /
loop; this policy:

1. pins and loads the converted episode (reusing :class:`DemoReplayPolicy`),
2. rebuilds fixed-anchor SE3 targets through :class:`FixedAnchorTargets`,
3. verifies the planner receipt scope with :mod:`scope`,
4. composes certified plans onto the 30 Hz grid through
   :class:`ManagedReplayWrapper`, and
5. returns a native23 command for each physics step.

No model forward, simulator step, IK solve, or custom control loop lives here.
The executor seam is injected (or registered remotely); an uncertified or
absent executor fails closed. Stdlib-only at import; torch and numpy are lazy.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Dict, List, Optional

from . import contracts, delta_tf, plan_wrapper, scope
from .errors import ContractViolation, Port2026Error
from .replay_policy import DemoReplayPolicy, ReplayExhausted, ReplayPinMismatch


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ManagedPolicyViolation(ContractViolation):
    """Managed policy construction or step violates the replay/seam contract."""


class ManagedReplayPolicy:
    """Evaluator policy: decoded replay targets -> certified plan -> native23.

    Args:
        converted_dir / manifest_sha256 / actions_sha256 / task_name /
        demo_id / instance_id: episode identity pins, forwarded to the inner
        :class:`DemoReplayPolicy`.
        anchors: fixed run-local start pose6 for left / right / chest.
        representation: saved-row representation (``pairwise_delta`` default).
        dt_seconds: saved control step dt (1/30).
        receipt_path: planner canary receipt JSON attached to this lane.
        expected_receipt_sha256 / expected_config_sha256: back-link pins.
        required_arm: arm scope this consumer needs (``left``/``right``/``both``).
        require_live_scene: whether a live-scene certificate is required.
        trace_path: JSONL trace path; falls back to ``BR_REPLAY_TRACE``.
        executor: optional duck-typed planner executor; remote registration
            fills the seam when absent.
        rows / manifest: test-only injection.
    """

    def __init__(
        self,
        converted_dir: Optional[str] = None,
        manifest_sha256: Optional[str] = None,
        actions_sha256: Optional[str] = None,
        anchors: Optional[Dict[str, List[float]]] = None,
        representation: str = delta_tf.REPRESENTATION_PAIRWISE,
        dt_seconds: float = contracts.CONTROL_DT,
        receipt_path: Optional[str] = None,
        expected_receipt_sha256: Optional[str] = None,
        expected_config_sha256: Optional[str] = None,
        required_arm: str = scope.ARM_LEFT,
        require_live_scene: bool = False,
        task_name: Optional[str] = None,
        demo_id: Optional[str] = None,
        instance_id: Optional[int] = None,
        trace_path: Optional[str] = None,
        executor: Any = None,
        rows: Optional[List[List[float]]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        receipt: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._inner = DemoReplayPolicy(
            converted_dir=converted_dir,
            manifest_sha256=manifest_sha256,
            actions_sha256=actions_sha256,
            start_frame=0,
            task_name=task_name,
            demo_id=demo_id,
            instance_id=instance_id,
            trace_path=trace_path,
            rows=rows,
            manifest=manifest,
        )
        if anchors is None:
            raise ManagedPolicyViolation("fixed-anchor anchors are required")
        self._targets = delta_tf.FixedAnchorTargets(
            rows=self._inner._rows,  # pin-validated rows from the inner policy
            anchors=anchors,
            dt_seconds=dt_seconds,
            representation=representation,
        )

        receipt: Dict[str, Any]
        if receipt is not None:
            receipt = dict(receipt)
        elif receipt_path:
            rp = pathlib.Path(receipt_path)
            if not rp.exists():
                raise ManagedPolicyViolation(f"planner receipt absent: {rp}")
            if expected_receipt_sha256 is not None:
                actual = _sha256_file(rp)
                if actual != expected_receipt_sha256:
                    raise ManagedPolicyViolation(
                        f"receipt sha drift: expected {expected_receipt_sha256}, got {actual}")
            receipt = json.loads(rp.read_text(encoding="utf-8"))
        else:
            receipt = {}
        self._scope = scope.verify_receipt(
            receipt,
            required_arm=required_arm,
            require_live_scene=require_live_scene,
            expected_receipt_sha=expected_receipt_sha256,
            expected_config_sha=expected_config_sha256,
        )
        self._scope.require()

        self._wrapper = plan_wrapper.ManagedReplayWrapper(
            targets=self._targets,
            scope_report=self._scope,
            executor=executor,
            physics_dt=dt_seconds,
        )
        self._executor_injected = executor is not None
        self._trace_path = trace_path or os.environ.get("BR_REPLAY_TRACE")
        self._action_dim: Optional[int] = None
        self._plan: Optional[Dict[str, Any]] = None
        self._step_in_plan = 0
        self._chunk_index = 0

    # -- evaluator seam -------------------------------------------------------

    def set_action_dim(self, action_dim: int) -> None:
        self._action_dim = int(action_dim)

    def reset(self) -> None:
        self._inner.reset()
        self._plan = None
        self._step_in_plan = 0
        self._chunk_index = 0
        self._emit("managed_policy_reset", {"index": self._chunk_index})

    def forward(self, obs: Optional[dict] = None, **kwargs: Any) -> Any:
        native23 = self._next_native23()
        import torch  # lazy; package import stays torch-free

        return torch.tensor(native23["vector"], dtype=torch.float32)

    act = forward

    # -- composition ----------------------------------------------------------

    def _next_native23(self) -> Dict[str, Any]:
        if self._plan is None:
            self._plan = self._wrapper.plan_row(self._chunk_index)
        command: Optional[Dict[str, Any]] = None
        try:
            command = self._wrapper.assemble_step(self._plan, self._step_in_plan)
        except plan_wrapper.planner_grid.TrajectoryExhausted:
            self._chunk_index += 1
            self._step_in_plan = 0
            self._plan = self._wrapper.plan_row(self._chunk_index)
            command = self._wrapper.assemble_step(self._plan, 0)
        self._step_in_plan += 1
        self._emit("managed_native23", {
            "chunk_index": self._chunk_index,
            "step_in_plan": self._step_in_plan - 1,
            "target_row": command["target_row"],
            "sources": command["sources"],
        })
        return command

    def bound_receipt(self) -> Dict[str, Any]:
        return {
            "policy": "ManagedReplayPolicy",
            "inner": self._inner.bound_receipt(),
            "scope": {"pass": self._scope.pass_, "missing": self._scope.missing,
                      "evidence": self._scope.evidence},
            "episode_length": self._wrapper.episode_length,
            "executor_injected": self._executor_injected,
            "evaluator_action_dim": self._action_dim,
        }

    # -- trace -----------------------------------------------------------------

    def _emit(self, event: str, detail: Dict[str, Any]) -> None:
        if not self._trace_path:
            return
        path = pathlib.Path(self._trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **detail}, allow_nan=False) + "\n")
            f.flush()
