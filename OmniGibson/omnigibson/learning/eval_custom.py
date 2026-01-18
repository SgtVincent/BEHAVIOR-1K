import csv
from inspect import getsourcefile
import json
import logging
import os
from pathlib import Path
import shutil
from signal import SIGINT
from signal import signal
import sys
import time
import traceback
from typing import Any

from av.container import Container
from av.stream import Stream
import cv2
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.robots.sim_robot.og_teleop_utils import augment_rooms
from gello.robots.sim_robot.og_teleop_utils import generate_robot_config
from gello.robots.sim_robot.og_teleop_utils import get_task_relevant_room_types
from gello.robots.sim_robot.og_teleop_utils import load_available_tasks
import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.pose_perturbator import PosePerturbator
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import HEAD_RESOLUTION
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from omnigibson.learning.utils.eval_utils import WRIST_RESOLUTION
from omnigibson.learning.utils.eval_utils import flatten_obs_dict
from omnigibson.learning.utils.eval_utils import generate_basic_environment_config
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.learning.utils.obs_utils import write_video
from omnigibson.macros import create_module_macros
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric
from omnigibson.metrics import MetricBase
from omnigibson.metrics import TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
import omnigibson.utils.transform_utils as T
import torch as th

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_EVAL_INSTANCES = 10
m.NUM_TRAIN_INSTANCES = 200


gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True


ROLLOUT_CAMERA_NAMES = [
    "head",
    "left_wrist",
    "right_wrist",
]


logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info


class Evaluator:
    """Evaluator class for running and evaluating policies for behavior task."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        self.env._current_episode = 0
        self._video_writer = None
        self._rollout_video_writers = None

        if self.cfg.perturb_pose:
            self._pose_perturbator = PosePerturbator(logger)
            np.random.seed(self.cfg.perturb_pose_seed)

        logger.info(f"{self.cfg=}")

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False
        env = og.Environment(configs=cfg)
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> BaseRobot:
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_policy(self) -> Any:
        policy = instantiate(self.cfg.model)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> list[MetricBase]:
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> tuple[bool, bool, dict]:
        self.robot_action = self.policy.forward(obs=self.obs)

        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            self.n_trials += 1

        for metric in self.metrics:
            metric.step_callback(self.env)
        return terminated, truncated, info

    @property
    def video_writer(self) -> tuple[Container, Stream]:
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writer = video_writer

    @property
    def rollout_video_writers(self) -> dict[str, tuple[Container, Stream]]:
        return self._rollout_video_writers

    @rollout_video_writers.setter
    def rollout_video_writers(self, rollout_video_writers: dict[str, tuple[Container, Stream]]) -> None:
        if self._rollout_video_writers is not None:
            for camera_name in ROLLOUT_CAMERA_NAMES:
                (container, stream) = self._rollout_video_writers[camera_name]
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
        self._rollout_video_writers = rollout_video_writers

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
            )
        with open(tro_file_path) as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)

                if self.cfg.perturb_pose:
                    perturbed_pos, perturbed_quat = self._pose_perturbator.perturb_robot_root_pose(
                        robot_pos, robot_quat
                    )
                    presampled_robot_poses[self.robot.model_name][0]["position"] = perturbed_pos
                    presampled_robot_poses[self.robot.model_name][0]["orientation"] = perturbed_quat

                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose)))
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (56, 56),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (56, 56),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (112, 112),
        )
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def _write_rollout(self) -> None:
        self.rollout_state_action["state"].append(self.obs["robot_r1::proprio"].cpu().numpy())
        self.rollout_state_action["action"].append(self.robot_action.cpu().numpy())

        for camera_name in ROLLOUT_CAMERA_NAMES:
            write_video(
                self.obs[ROBOT_CAMERA_NAMES["R1Pro"][camera_name] + "::rgb"].numpy()[None, ...],
                video_writer=self.rollout_video_writers[camera_name],
                batch_size=1,
                mode="rgb",
            )

    def success_callback(self, success: bool) -> None:
        if success:
            self.n_success_trials += 1
            sucess_video_name = self.cur_video_name.replace(".mp4", "_success.mp4")
            try:
                shutil.move(self.cur_video_name, sucess_video_name)
            except Exception:
                logger.warning(f"Failed to move video {self.cur_video_name} to {sucess_video_name}")
            self.cur_video_name = sucess_video_name
            if hasattr(self, "rollout_paths"):
                np.savez_compressed(self.rollout_paths["state_action"], self.rollout_state_action)
                logger.info(f"Saved rollout data to {self.rollout_paths['state_action']}")
        elif hasattr(self, "rollout_paths"):
            for camera_name in ROLLOUT_CAMERA_NAMES:
                try:
                    os.remove(self.rollout_paths[camera_name])
                except Exception:
                    logger.warning(f"Failed to remove rollout video {self.rollout_paths.get(camera_name)}")

    def reset(self) -> None:
        self.obs = self._preprocess_obs(self.env.reset()[0])
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.rollout_video_writers = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()

    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda: 0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("openpi-comet.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)

    gm.HEADLESS = config.headless

    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    if config.save_rollout:
        rollout_path = Path(config.log_path).expanduser() / "rollouts"
        rollout_path.mkdir(parents=True, exist_ok=True)

    assert not (
        config.eval_on_train_instances and config.test_hidden
    ), "Cannot eval on train instances and test hidden instances simultaneously."
    if config.test_hidden:
        logger.info("You are evaluating on hidden test instances! This is for internal use only.")

    if config.eval_on_train_instances:
        logger.info("You are evaluating on training instances, set eval_on_train_instances to False for test instances.")
        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f]
        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(str(int((episode["episode_index"] // 10) % 1e3)))
        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(set(range(m.NUM_TRAIN_INSTANCES))), (
                f"eval instance ids must be in range({m.NUM_TRAIN_INSTANCES})"
            )
            instances_to_run = [instances_to_run[i] for i in config.eval_instance_ids]
    elif config.test_hidden:
        instances_to_run = config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))
        assert set(instances_to_run).issubset(set(range(m.NUM_EVAL_INSTANCES))), (
            f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
        )
    else:
        if config.use_parallel_evaluator:
            instances_to_run = set(range(config.parallel_evaluator_start_idx, config.parallel_evaluator_end_idx))
            logger.info(
                f"Using parallel evaluator with start index {config.parallel_evaluator_start_idx} and end index {config.parallel_evaluator_end_idx}"
            )
        else:
            instances_to_run = config.eval_instance_ids if config.eval_instance_ids is not None else set(range(m.NUM_EVAL_INSTANCES))

        assert set(instances_to_run).issubset(set(range(m.NUM_EVAL_INSTANCES))), (
            f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
        )
        task_instance_csv_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv")
        with open(task_instance_csv_path) as f:
            lines = list(csv.reader(f))[1:]
        assert lines[TASK_NAMES_TO_INDICES[config.task.name]][1] == config.task.name, (
            f"Task name from config {config.task.name} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
        )
        test_instances = lines[TASK_NAMES_TO_INDICES[config.task.name]][2].strip().split(",")
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]

    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        for epi in range(m.NUM_EVAL_EPISODES):
            evaluator.reset()
            for idx in instances_to_run:
                evaluator.reset()
                evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
                logger.info(f"Starting task instance {idx} / episode {epi} for evaluation...")

                done = False
                if config.write_video:
                    evaluator.cur_video_name = f"{video_path!s}/{config.task.name}_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=evaluator.cur_video_name,
                        resolution=(112, 168),
                    )

                if config.save_rollout:
                    rollout_video_writers = {}
                    rollout_paths = {}
                    for camera_name in ROLLOUT_CAMERA_NAMES:
                        rollout_id_path = Path(rollout_path) / f"{int(idx):04d}_{int(epi):04d}"
                        rollout_id_path.mkdir(parents=True, exist_ok=True)
                        rollout_paths[camera_name] = str(rollout_id_path / f"{camera_name}.mp4")
                        rollout_video_writers[camera_name] = create_video_writer(
                            fpath=rollout_paths[camera_name],
                            resolution=HEAD_RESOLUTION if camera_name == "head" else WRIST_RESOLUTION,
                        )

                    evaluator.rollout_video_writers = rollout_video_writers
                    rollout_paths["state_action"] = str(rollout_id_path / "state_action.npz")
                    evaluator.rollout_paths = rollout_paths
                    evaluator.rollout_state_action = {"state": [], "action": []}
                    logger.info(f"created rollout video writers and saved rollout video to {rollout_id_path}")

                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                while not done:
                    time_start = time.time()
                    terminated, truncated, info = evaluator.step()
                    time_step = time.time() - time_start

                    if time_step > 15 * 60:
                        logger.error(f"Step timeout: {time_step} seconds, terminating evaluation")
                        exit(1)

                    if terminated or truncated:
                        done = True
                    if config.save_rollout:
                        evaluator._write_rollout()
                    if config.write_video and evaluator.env._current_step % 20 == 0:
                        evaluator._write_video()
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")

                    if terminated or truncated:
                        evaluator.success_callback(info["done"]["success"])

                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: {terminated}, {truncated}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())
                with open(metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w") as f:
                    json.dump(metrics, f)
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {evaluator.cur_video_name}")
                if config.save_rollout:
                    evaluator.rollout_video_writers = None
                    evaluator.rollout_paths = None
                    evaluator.rollout_state_action = None
                    logger.info(f"Saved rollout video to {evaluator.rollout_paths}")
                else:
                    logger.warning("No observations were recorded.")