"""Episode operator: the real MoMa caller over the 2026 SDK boundary.

Role and boundaries
--------------------

This orchestrates one single-environment episode against the verified post1
surface:

- environment lifecycle: reset, step, BDDL success read;
- robot/observer reads: assisted-grasping records are consumed from the live
  SDK accessor after every step, never replayed from a source npz;
- boundary decision: privileged S2 ``ChatCompletions`` client;
- execution: the seam executor returns a tagged ``native23_joint`` command
  that is forwarded through the real environment step.

Immutability rules (a28 manifest):

- The a28 raw initial state may be restored **once** before the first action
  for same-source comparison. After episode start, reset / load_state /
  set_joint_positions / teleport are forbidden and raise.
- A zero-action global-env smoke run is not evidence of a non-zero or
  three-can closed loop; the operator only records what the live consumer
  actually read and stepped.

The operator does not import omnigibson/torch: live objects arrive through the
boundary adapters and injected callables, which keeps the orchestration
CPU-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .sdk.env_adapter import SdkEnvironmentAdapter
from .sdk.robot_accessor import SdkRobotAccessor
from .s2.contract import RequestTracker, digest, full_task_terminal
from .schema import TaggedAction


class EpisodeImmutableAfterStart(RuntimeError):
    """Post-start state mutation (reset/load_state/teleport) was attempted."""


class ExecutorAdapterMissing(RuntimeError):
    """No MP executor is bound. The operator refuses to plan or env.step."""


class ExecutorPlanRejected(RuntimeError):
    """The bound executor returned a rejected/invalid plan.

    Raised when the executor raises, omits the tagged action, omits the
    certificate, returns a non-tagged action, or returns an all-zero
    placeholder command. A rejected plan must never be turned into a
    zero-action env.step; the caller records the rejection instead.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class EpisodeConfig:
    run_id: str
    can_names: List[str]
    bucket_name: str
    max_steps: int
    step_hz: int = 30
    physics_hz: int = 120
    allow_initial_state_restore: bool = True


@dataclass
class StepRecord:
    step: int
    action_label: Optional[str]
    action_dim: Optional[int]
    events: List[Dict[str, Any]] = field(default_factory=list)
    arms: Dict[str, Any] = field(default_factory=dict)
    certificate: Dict[str, Any] = field(default_factory=dict)
    s2: Dict[str, Any] = field(default_factory=dict)
    terminal: bool = False


class MomaEpisodeOperator:
    """Single-env MoMa caller: reset -> observe -> boundary decide -> step -> record."""

    def __init__(
        self,
        cfg: EpisodeConfig,
        env_adapter: SdkEnvironmentAdapter,
        robot_accessor: SdkRobotAccessor,
        *,
        observe_fn: Callable[..., Dict[str, Any]],
        transition_fn: Callable[..., List[Dict[str, Any]]],
        executor: Any,
        s2_client: Any = None,
        tracker: Optional[RequestTracker] = None,
        resolve_objects: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        live_reads: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.env = env_adapter
        self.robot = robot_accessor
        self.observe_fn = observe_fn
        self.transition_fn = transition_fn
        self.executor = executor
        self.s2_client = s2_client
        self.tracker = tracker or RequestTracker(cfg.run_id)
        self.resolve_objects = resolve_objects
        self.live_reads = live_reads or {}

        self._started = False
        self._initial_load_done = False
        self._step = 0
        self._previous_obs: Optional[Dict[str, Any]] = None
        self._recent_turns: List[Dict[str, Any]] = []
        self.records: List[StepRecord] = []

    # -- lifecycle ---------------------------------------------------------

    def reset_episode(
        self,
        *,
        perform_initial_load: bool = False,
        load_state_fn: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """Reset the episode and observer/grasp bookkeeping.

        ``perform_initial_load=True`` restores the frozen a28 initial state
        once through ``load_state_fn``. A second reset or load after episode
        start raises :class:`EpisodeImmutableAfterStart`.
        """
        if self._started:
            raise EpisodeImmutableAfterStart("episode already started; post-start reset forbidden")
        if perform_initial_load:
            if not self.cfg.allow_initial_state_restore:
                raise EpisodeImmutableAfterStart("initial-state restore disabled by config")
            if self._initial_load_done:
                raise EpisodeImmutableAfterStart("initial-state restore allowed only once")
            if load_state_fn is None:
                raise ValueError("load_state_fn required for initial restore")
            load_state_fn()
            self._initial_load_done = True

        obs = self.env.reset()
        self._step = 0
        self._previous_obs = None
        self._recent_turns = []
        self.records = []
        self.tracker.pending = None
        self.tracker.completed = []
        return obs

    def restore_initial_state(self, load_state_fn: Callable[[], Any]) -> None:
        """One-time legal same-source initial restore; teleport after start rejected."""
        if self._started:
            raise EpisodeImmutableAfterStart("post-start state restore forbidden")
        if self._initial_load_done:
            raise EpisodeImmutableAfterStart("initial-state restore allowed only once")
        if not self.cfg.allow_initial_state_restore:
            raise EpisodeImmutableAfterStart("initial-state restore disabled by config")
        load_state_fn()
        self._initial_load_done = True

    def forbid_post_start_mutation(self) -> None:
        """Hook for the runner to guard robot.set_joint_positions / reset paths."""
        if self._started:
            raise EpisodeImmutableAfterStart("post-start state mutation forbidden")

    # -- observation ------------------------------------------------------

    def _resolve_scene_objects(self) -> Dict[str, Any]:
        scope = self.env.object_scope()
        if self.resolve_objects is not None:
            return self.resolve_objects(scope)
        return scope

    def observe(self) -> Dict[str, Any]:
        scope = self._resolve_scene_objects()
        missing_cans = [name for name in self.cfg.can_names if name not in scope]
        bucket = scope.get(self.cfg.bucket_name)

        # Missing scene objects must surface as BLOCKED observation entries,
        # never a bare KeyError and never a zero-filled placeholder.
        if missing_cans or bucket is None:
            obs: Dict[str, Any] = {
                "objects": {},
                "task_progress": {},
                "runner_state": "blocked",
                "decision_required": False,
                "active_request": None,
            }
            for name in missing_cans:
                obs["objects"][name] = {"status": "BLOCKED", "error": "object_not_in_scope", "object": name}
            if bucket is None:
                obs["objects"][self.cfg.bucket_name] = {
                    "status": "BLOCKED", "error": "object_not_in_scope",
                    "object": self.cfg.bucket_name,
                }
            arms: Dict[str, Any] = {}
            for arm in self.robot.arm_names():
                arms[arm] = self.robot.safe_hand_record(arm)
            obs["robot_state"] = {"arms": arms}
            return obs

        cans = {name: scope[name] for name in self.cfg.can_names}
        obs = self.observe_fn(
            self.robot.robot,
            cans,
            bucket,
            pose_fn=self.live_reads.get("pose_fn"),
            inside_type=self.live_reads.get("inside_type"),
            contact_type=self.live_reads.get("contact_type"),
        )
        arms = {}
        for arm in self.robot.arm_names():
            arms[arm] = self.robot.safe_hand_record(arm)
        obs.setdefault("robot_state", {})["arms"] = arms
        return obs

    def _events(self, current: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._previous_obs is None:
            return []
        return self.transition_fn(self._previous_obs, current, self.cfg.can_names)

    @staticmethod
    def _validate_plan(plan: Any) -> tuple[TaggedAction, Dict[str, Any]]:
        """Reject an invalid/empty plan before any env.step is issued.

        The seam contract: ``{"action23": TaggedAction, "certificate": dict}``.
        An all-zero vector is treated as a placeholder, not a solved command.
        """
        if not isinstance(plan, dict):
            raise ExecutorPlanRejected("plan_not_dict", f"got {type(plan).__name__}")
        if "action23" not in plan:
            raise ExecutorPlanRejected("action23_missing", "plan has no action23")
        action = plan["action23"]
        if not isinstance(action, TaggedAction):
            raise ExecutorPlanRejected(
                "action23_not_tagged", f"got {type(action).__name__}"
            )
        certificate = plan.get("certificate")
        if not isinstance(certificate, dict):
            raise ExecutorPlanRejected("certificate_missing", "certificate must be a dict")
        if all(float(v) == 0.0 for v in action.to_list()):
            raise ExecutorPlanRejected(
                "zero_action_placeholder", "action23 vector is all zero"
            )
        return action, certificate

    # -- decision / step ---------------------------------------------------

    def request_decision(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.s2_client is None:
            raise ValueError("s2_client not bound; no privileged decision available")
        decision = self.s2_client.decide(obs, self._recent_turns)
        self._recent_turns.append({"messages": [{"role": "user", "content": obs}]})
        return decision

    def step(self) -> StepRecord:
        """Observe, decide at the boundary, execute one tagged command, record."""
        if not self._started:
            self._started = True

        current = self.observe()
        events = self._events(current)
        arms = dict(current.get("robot_state", {}).get("arms", {}))

        # Boundary short-circuit: a blocked / non-decision observation must not
        # reach the S2 HTTP caller or the executor. No API request, no env.step.
        if current.get("runner_state") == "blocked" or current.get("decision_required") is not True:
            events = events + [{
                "type": "boundary_blocked",
                "reason": current.get("runner_state") or "decision_not_required",
            }]
            record = StepRecord(step=self._step, action_label=None, action_dim=None,
                                events=events, arms=arms)
            self._previous_obs = current
            self.records.append(record)
            return record

        terminal = full_task_terminal(current, self._s2_cfg())
        record = StepRecord(step=self._step, action_label=None, action_dim=None, events=events, arms=arms)

        if terminal:
            record.terminal = True
            self._previous_obs = current
            self.records.append(record)
            return record

        if self.executor is None:
            raise ExecutorAdapterMissing(
                "MP executor adapter missing; no plan and no env.step issued"
            )

        decision = self.request_decision(current)
        try:
            plan = self.executor.plan(decision["args"])
        except ExecutorAdapterMissing:
            raise
        except Exception as exc:
            raise ExecutorPlanRejected(
                "executor_raised", f"{type(exc).__name__}: {exc}"
            ) from exc
        action, certificate = self._validate_plan(plan)
        record.action_label = action.label
        record.action_dim = len(action.vector)
        record.certificate = certificate
        record.s2 = {
            "call_id": decision.get("call_id"),
            "latency_s": decision.get("latency_s"),
            "request_sha256": decision.get("request_sha256"),
        }

        self.env.step(action.to_list())
        self._step += 1
        self._previous_obs = current
        self.records.append(record)
        return record

    def _s2_cfg(self) -> Dict[str, Any]:
        return {
            "can_names": self.cfg.can_names,
            "bucket_name": self.cfg.bucket_name,
            **self.live_reads.get("s2_cfg", {}),
        }

    # -- terminal read -----------------------------------------------------

    def bddl_success(self) -> bool:
        return self.env.task_success()

    def closed_loop_success(self, current: Optional[Dict[str, Any]] = None) -> bool:
        """BDDL success AND the a28 full terminal gate; never bool(dict) alone."""
        if not self.bddl_success():
            return False
        obs = current if current is not None else self.observe()
        return full_task_terminal(obs, self._s2_cfg())

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def finished(self) -> bool:
        return self._step >= self.cfg.max_steps or any(rec.terminal for rec in self.records)

    def observation_digest(self, obs: Dict[str, Any]) -> str:
        return digest(obs)
