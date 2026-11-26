"""Concrete S1/S2 episode runner over the 2026 SDK boundary.

This is the production caller entry ported from the a28 frozen
``pnp_object_bridge.py`` main loop and ``s2_runtime.py`` frontend contract:

- ``prepare`` wires the config, SDK adapters, observer, S2 client, step writer
  and the manifest-injected MP executor; missing pieces are reported, never
  replaced by a zero-action fallback;
- ``run_episode`` resets the episode (one legal same-source initial restore),
  observes, asks the privileged S2 boundary, executes the tagged command
  through the real environment step, records every row, then finalizes the
  closed-loop report;
- the cuRobo v2 MP executor arrives through the integrator-owned plugin
  manifest; until it resolves, ``prepare`` reports the plugin blocker and the
  runner does not start a physics episode.

Sim-free: the runner itself imports no simulator. Live env/robot objects are
injected through the SDK adapters; heavy simulator coupling stays behind the
SimAdapter seam for the GPU-bound canary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import RunnerConfig
from .episode_writer import EpisodeWriter
from .mp_plugin import PluginUnavailable, classify_plugin, load_executor, PluginSpec
from .operator import MomaEpisodeOperator, EpisodeConfig
from .s2.client import ChatCompletionsClient
from .sdk.env_adapter import SdkEnvironmentAdapter
from .sdk.robot_accessor import SdkRobotAccessor


@dataclass
class PrepareReport:
    ready: bool
    config_digest: str
    bound: List[str] = field(default_factory=list)
    deferred: List[str] = field(default_factory=list)
    blockers: List[Dict[str, str]] = field(default_factory=list)
    plugin: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "config_digest": self.config_digest,
            "bound": list(self.bound),
            "deferred": list(self.deferred),
            "blockers": list(self.blockers),
            "plugin": dict(self.plugin),
        }


class MomaRunner:
    """Single-env three-can runner: prepare -> run episode -> finalize."""

    def __init__(
        self,
        cfg: RunnerConfig,
        *,
        env_adapter: Optional[SdkEnvironmentAdapter] = None,
        robot_accessor: Optional[SdkRobotAccessor] = None,
        observer_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        transition_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        executor: Any = None,
        s2_client: Optional[ChatCompletionsClient] = None,
        writer: Optional[EpisodeWriter] = None,
        live_reads: Optional[Dict[str, Any]] = None,
        plugin_manifest: Optional[Dict[str, Any]] = None,
        prompt_text: str = "",
    ) -> None:
        self.cfg = cfg
        self.env_adapter = env_adapter or SdkEnvironmentAdapter()
        self.robot_accessor = robot_accessor or SdkRobotAccessor()
        self.observer_fn = observer_fn
        self.transition_fn = transition_fn
        self.executor = executor
        self.s2_client = s2_client
        self.writer = writer
        self.live_reads = live_reads or {}
        self.plugin_manifest = plugin_manifest
        self.prompt_text = prompt_text
        self.operator: Optional[MomaEpisodeOperator] = None

    # -- S2 client factory --------------------------------------------------

    def build_s2_client(self, opener: Any = None, audit_dir: Optional[Path] = None) -> ChatCompletionsClient:
        return ChatCompletionsClient(
            base_url=self.cfg.model_base_url,
            model=self.cfg.model_name,
            prompt_text=self.prompt_text,
            cfg=self.cfg.s2_cfg(),
            history_turns=self.cfg.history_turns,
            timeout_s=self.cfg.model_timeout_s,
            max_response_bytes=self.cfg.model_max_response_bytes,
            temperature=self.cfg.model_temperature,
            api_key_env=self.cfg.model_api_key_env,
            opener=opener,
            audit_dir=audit_dir,
        )

    # -- prepare -----------------------------------------------------------

    def prepare(self) -> PrepareReport:
        problems = self.cfg.validate()
        blockers: List[Dict[str, str]] = []
        for problem in problems:
            blockers.append({"kind": "config", "detail": problem})

        deferred: List[str] = []
        bound: List[str] = []

        if self.observer_fn is None:
            blockers.append({"kind": "observer", "detail": "observer_fn not injected"})
        else:
            bound.append("observer")
        if self.transition_fn is None:
            blockers.append({"kind": "observer", "detail": "transition_fn not injected"})
        else:
            bound.append("transitions")

        if self.env_adapter.bound:
            bound.append("env_adapter")
        else:
            deferred.append("env_adapter (bound in GPU canary through SimAdapter)")
        if self.robot_accessor.bound:
            bound.append("robot_accessor")
        else:
            deferred.append("robot_accessor (bound in GPU canary through SDK adapter)")

        # Executor: injected instance wins; otherwise resolve the plugin manifest.
        plugin_report: Dict[str, Any] = {}
        if self.executor is None:
            if self.plugin_manifest is None:
                blockers.append({"kind": "executor", "detail": "no executor and no plugin manifest"})
            else:
                try:
                    spec = PluginSpec.from_manifest(self.plugin_manifest)
                    plugin_report = classify_plugin(spec)
                    if plugin_report["status"] == "OK":
                        context = {"cfg": self.cfg, "robot": self.robot_accessor}
                        self.executor = load_executor(self.plugin_manifest, context=context)
                        bound.append("executor_plugin")
                    else:
                        blockers.append({"kind": "executor",
                                         "detail": f"plugin unavailable: {plugin_report['detail']}"})
                except PluginUnavailable as exc:
                    plugin_report = {"status": "MISSING", "detail": f"{exc.reason}: {exc.detail}"}
                    blockers.append({"kind": "executor", "detail": plugin_report["detail"]})
        else:
            bound.append("executor_injected")

        if self.s2_client is None:
            if self.cfg.policy_mode == "external_s2":
                if not self.prompt_text:
                    blockers.append({"kind": "s2", "detail": "prompt_text required for external_s2"})
                else:
                    deferred.append("s2_client (built at run boundary; opener injectable for CPU tests)")
            else:
                deferred.append("s2_client internal policy mode")
        else:
            bound.append("s2_client")

        if self.writer is None:
            out_dir = Path(self.cfg.output_dir or ".") / self.cfg.run_id
            self.writer = EpisodeWriter(out_dir)
        bound.append("writer")

        ready = not blockers
        report = PrepareReport(
            ready=ready,
            config_digest=self.cfg.effective_digest(),
            bound=bound,
            deferred=deferred,
            blockers=blockers,
            plugin=plugin_report,
        )

        if ready:
            episode_cfg = EpisodeConfig(
                run_id=self.cfg.run_id,
                can_names=list(self.cfg.can_names),
                bucket_name=self.cfg.bucket_name,
                max_steps=self.cfg.max_steps,
                step_hz=self.cfg.step_hz,
                physics_hz=self.cfg.physics_hz,
                allow_initial_state_restore=True,
            )
            self.operator = MomaEpisodeOperator(
                episode_cfg,
                self.env_adapter,
                self.robot_accessor,
                observe_fn=self.observer_fn,
                transition_fn=self.transition_fn,
                executor=self.executor,
                s2_client=self.s2_client,
                live_reads=self.live_reads,
            )
        return report

    # -- episode -----------------------------------------------------------

    def run_episode(self, *, load_state_fn: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
        if self.operator is None:
            raise RuntimeError("prepare() did not produce a bound operator; resolve blockers first")
        assert self.writer is not None
        op = self.operator

        op.reset_episode(perform_initial_load=load_state_fn is not None, load_state_fn=load_state_fn)

        issued = 0
        while not op.finished:
            record = op.step()
            issued += 1
            row: Dict[str, Any] = {
                "step": record.step,
                "action_label": record.action_label,
                "action_dim": record.action_dim,
                "events": record.events,
                "arms": record.arms,
                "certificate": record.certificate,
                "s2": record.s2,
                "terminal": record.terminal,
            }
            self.writer.record_step(row, flush=(issued == 1 or issued % 30 == 0 or bool(record.events)))
            if record.terminal:
                break

        success = op.closed_loop_success()
        report: Dict[str, Any] = {
            "run_id": self.cfg.run_id,
            "task_name": self.cfg.task_name,
            "policy_mode": self.cfg.policy_mode,
            "assisted_grasping": self.cfg.assisted_grasping,
            "issued_actions": issued,
            "closed_loop_success": bool(success),
            "bddl_success": op.bddl_success(),
            "config_digest": self.cfg.effective_digest(),
            "write_video": self.cfg.write_video,
        }
        return self.writer.finalize(report)
