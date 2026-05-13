import logging
import torch as th
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from typing import Optional


__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
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
            return self.last_action
        # convert observation to numpy
        obs = torch_to_numpy(obs)
        self.last_action = self.policy.act(obs).detach().cpu()
        return self.last_action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.last_action = None
