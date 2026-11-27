"""Saved delta-TF policy using upstream policy / evaluator base classes.

The official evaluator owns env.step, metrics and video. This module does not
implement a parallel evaluation loop or a websocket / model-serving stack.
"""
import numpy as np
import torch as th
from hydra.utils import instantiate

from omnigibson.action_primitives.delta_tf_action_primitives import DeltaTFActionPrimitives
from omnigibson.eval.evaluator import BatchedEvaluator
from omnigibson.eval.policies import LocalPolicy


class DeltaTFReplayPolicy(LocalPolicy):
    """Tracking-only, fixed-base replay adapter; not full BRVLA admission.

    npz_path and array_key explicitly select already converted execution23 data.
    Initial-state restore and producer/checkpoint lineage belong to the caller.
    Finite exhaustion raises StopIteration rather than repeating source rows or
    returning zero joint targets. The evaluator must mark it as a diagnostic
    exit, not a task-success result.
    """

    def __init__(self, env, robot, npz_path, array_key, tool_links, representation, **kwargs):
        super().__init__(action_dim=robot.action_dim)
        with np.load(npz_path, allow_pickle=False) as data:
            self.rows = np.array(data[array_key], copy=True)
        if self.rows.ndim != 2 or self.rows.shape[1] != 23 or len(self.rows) == 0:
            raise ValueError("Converted data must be a nonempty Nx23 array")
        if not np.isfinite(self.rows).all():
            raise ValueError("Converted data contains non-finite entries")
        self.primitives = DeltaTFActionPrimitives(env, robot, tool_links, representation=representation)
        self.reset()

    def reset(self):
        self._index = 0
        self._generator = None
        self._anchored = False

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
            return action.unsqueeze(0)


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
