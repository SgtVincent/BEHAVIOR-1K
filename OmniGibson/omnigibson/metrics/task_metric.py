import numpy as np
import omnigibson as og
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional


class TaskMetric(MetricBase):
    def __init__(self, human_stats: Optional[dict] = None):
        self.timesteps = 0
        self.human_stats = human_stats
        if human_stats is None:
            print("No human stats provided.")
        else:
            self.human_stats = {
                "steps": self.human_stats["length"],
            }

    def start_callback(self, env):
        self.timesteps = 0
        self.render_timestep = og.sim.get_rendering_dt()

        # --- ADDITIVE (ours): RLinf-comparable `success_once` accounting ------------------
        # RLinf's BEHAVIOR eval reports `eval/success_once`, which OR-accumulates the
        # instantaneous success flag across EVERY step of the episode
        #   RLinf/rlinf/envs/behavior/behavior_env.py:765
        #     success_once[i] = success_once[i] | done_dict.get("success", False)
        # The challenge metric `q_score.final` (end_callback below) is instead evaluated
        # ONLY at the terminal state, and the underlying flag does NOT latch
        #   OmniGibson/omnigibson/tasks/task_base.py:363  ->  self._success = ...  (plain `=`)
        # so strictly success_once >= (q_score.final == 1.0). Recording both lets a single
        # episode yield BOTH conventions, so our numbers can be placed next to RLinf's
        # without a criterion mismatch.
        # This block is PURELY ADDITIVE: it never reads or writes `final_q_score`.
        self.success_once = False
        self.first_success_step = None
        self.success_step_count = 0
        # ---------------------------------------------------------------------------------

        # Store the initial state (true/false) of each predicate for each option
        self.initial_predicate_states = [
            [pred.evaluate() for pred in option] for option in env.task.ground_goal_state_options
        ]

    def step_callback(self, env):
        self.timesteps += 1

        # --- ADDITIVE (ours): mirror RLinf's OR-accumulation; see start_callback ----------
        try:
            step_success = bool(env.task.success)
        except Exception:
            # task.success asserts at least one step has occurred; treat as not-yet-known.
            step_success = False
        if step_success:
            self.success_step_count += 1
            if self.first_success_step is None:
                self.first_success_step = self.timesteps
            self.success_once = True
        # ---------------------------------------------------------------------------------

    def end_callback(self, env):
        # If task is fully complete, return perfect score
        if env.task.success:
            self.final_q_score = 1.0
            return

        # Otherwise calculate partial credit based on newly satisfied predicates. The partial credit is the maximum progress
        # made, across any of the groundings, from the initial state of the task.
        self.final_q_score = max(
            sum(
                int(not initially_true and pred.evaluate())
                for pred, initially_true in zip(option, option_previous_state)
            )
            / len(option)
            for option, option_previous_state in zip(env.task.ground_goal_state_options, self.initial_predicate_states)
        )

    def gather_results(self):
        return {
            "q_score": {"final": self.final_q_score},
            # --- ADDITIVE (ours): RLinf-comparable criterion; see start_callback ----------
            # `success_once` == RLinf's `eval/success_once` semantics (OR over all steps).
            # `first_success_step` / `success_step_count` quantify the gap between the two
            # conventions: a nonzero count with q_score.final < 1.0 is exactly the
            # "achieved at some point but did not hold to the end" case.
            "success_once": bool(getattr(self, "success_once", False)),
            "success_detail": {
                "first_success_step": getattr(self, "first_success_step", None),
                "success_step_count": getattr(self, "success_step_count", 0),
            },
            # -----------------------------------------------------------------------------
            "time": {
                "simulator_steps": self.timesteps,
                "simulator_time": self.timesteps * self.render_timestep,
                "normalized_time": self.human_stats["steps"] / self.timesteps,
            },
        }
