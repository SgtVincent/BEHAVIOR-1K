import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch as th
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy


__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
    "DemoActionReplayPolicy",
]


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(self, *args, action_dim: Optional[int] = None, **kwargs) -> None:
        self.policy = None  # To be set later
        self.action_dim = action_dim

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            return th.zeros(self.action_dim, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        expected_task_name: Optional[str] = None,
        expected_task_prompt_sha256: Optional[str] = None,
        expected_server_run_id: Optional[str] = None,
        expected_server_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.last_action = None
        self.policy = None
        self._expected_task_name = expected_task_name
        self._expected_task_prompt_sha256 = expected_task_prompt_sha256
        self._expected_server_run_id = expected_server_run_id
        self._expected_server_token = expected_server_token
        if host is not None or port is not None:
            self.policy = WebsocketClientPolicy(
                host=host,
                port=port,
                expected_task_name=expected_task_name,
                expected_task_prompt_sha256=expected_task_prompt_sha256,
                expected_server_run_id=expected_server_run_id,
                expected_server_token=expected_server_token,
            )

    def update_host(self, host: str, port: int) -> None:
        self.policy = WebsocketClientPolicy(
            host=host,
            port=port,
            expected_task_name=self._expected_task_name,
            expected_task_prompt_sha256=self._expected_task_prompt_sha256,
            expected_server_run_id=self._expected_server_run_id,
            expected_server_token=self._expected_server_token,
        )

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if "need_new_action" in obs and not obs["need_new_action"] and self.last_action is not None:
            if hasattr(self.policy, "cached_actions_remaining") and getattr(self.policy, "cached_actions_remaining") > 0:
                self.last_action = self.policy.pop_cached_action()
                return self.last_action
            return self.last_action
        # convert observation to numpy
        obs = torch_to_numpy(obs)
        self.last_action = self.policy.act(obs).detach().cpu()
        return self.last_action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.last_action = None


class DemoActionReplayPolicy:
    """Policy that replays actions from a BEHAVIOR demo parquet.

    This is intended for diagnostics / teacher-forcing checks, e.g. verifying
    that state restoration plus the primitive success checker can reproduce a
    demo segment when fed the recorded GT actions.
    """

    def __init__(
        self,
        *args,
        demo_data_path: str,
        demo_id: str,
        task_id: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        action_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.demo_data_path = Path(demo_data_path).expanduser()
        self.demo_id = str(demo_id).zfill(8)
        self.task_id = int(task_id) if task_id is not None else int(self.demo_id) // 100000
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame) if end_frame is not None else None
        self.action_dim = action_dim
        self._step = 0

        self._load_demo_data(self.demo_id, self.task_id)

    def _load_demo_data(self, demo_id: str, task_id: int) -> None:
        """Load / reload the backing parquet for demo action replay."""

        self.demo_id = str(demo_id).zfill(8)
        self.task_id = int(task_id)
        parquet_path = self.demo_data_path / "data" / f"task-{self.task_id:04d}" / f"episode_{self.demo_id}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Demo parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        if "action" not in df.columns:
            raise KeyError(f"Missing 'action' column in {parquet_path}")
        self.actions = np.asarray(df["action"].tolist(), dtype=np.float32)
        if self.actions.ndim != 2:
            raise ValueError(f"Expected 2D action array, got shape={self.actions.shape}")
        if self.action_dim is None:
            self.action_dim = int(self.actions.shape[1])
        logging.info(
            "Loaded demo action replay policy: demo=%s task_id=%d frames=%d start=%d end=%s action_dim=%d",
            self.demo_id,
            self.task_id,
            len(self.actions),
            self.start_frame,
            self.end_frame,
            self.action_dim,
        )

    def load_demo(self, demo_id: str, task_id: Optional[int] = None) -> None:
        """Reload action data when a persistent evaluator switches demos."""

        task_id = int(task_id) if task_id is not None else int(str(demo_id).zfill(8)) // 100000
        demo_id = str(demo_id).zfill(8)
        if demo_id != self.demo_id or int(task_id) != int(self.task_id):
            self._load_demo_data(demo_id, int(task_id))
        else:
            self._step = 0

    def reset(self) -> None:
        self._step = 0

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        frame_idx = self.start_frame + self._step
        if self.end_frame is not None:
            frame_idx = min(frame_idx, self.end_frame - 1)
        frame_idx = max(0, min(frame_idx, len(self.actions) - 1))
        action = self.actions[frame_idx]
        self._step += 1
        return th.as_tensor(action, dtype=th.float32)

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)
