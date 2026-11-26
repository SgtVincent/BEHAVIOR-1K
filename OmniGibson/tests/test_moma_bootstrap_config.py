"""CPU tests for the source-compat bootstrap config builder (sim-free).

Fixtures mirror the sealed source episode config dumped by probe 128; the
post1 schema constants come from the read-only evidence recorded in
``bootstrap_config``. No simulator, torch or h5py is imported here.
"""
from __future__ import annotations

from omnigibson.moma.bootstrap_config import (
    TARGET_CONTROLLER_ORDER,
    TARGET_PROPRIO_KEYS,
    TARGET_RESET_COUNT,
    CompatConfigBuilder,
    validate_effective,
)


def _controller_config() -> dict:
    return {
        "arm_left": {"name": "JointController", "motor_type": "position",
                     "pos_kp": 150, "command_input_limits": None,
                     "command_output_limits": None, "use_impedances": False,
                     "use_delta_commands": False},
        "arm_right": {"name": "JointController", "motor_type": "position",
                      "pos_kp": 150, "command_input_limits": None,
                      "command_output_limits": None, "use_impedances": False,
                      "use_delta_commands": False},
        "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth",
                         "command_input_limits": "default",
                         "command_output_limits": "default"},
        "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth",
                          "command_input_limits": "default",
                          "command_output_limits": "default"},
        "base": {"name": "HolonomicBaseJointController", "motor_type": "velocity",
                 "vel_kp": 150, "command_input_limits": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 "command_output_limits": [[-0.75, -0.75, -1.0], [0.75, 0.75, 1.0]],
                 "use_impedances": False},
        "trunk": {"name": "JointController", "motor_type": "position",
                  "pos_kp": 150, "command_input_limits": None,
                  "command_output_limits": None, "use_impedances": False,
                  "use_delta_commands": False},
        "camera": {"name": "NullJointController"},
    }


def _reset() -> list:
    return [0.0] * 6 + [1.025, -1.45, -0.47, 0.0] + [0.0] * 14 + [0.05] * 4


def _source_cfg(**overrides) -> dict:
    cfg = {
        "env": {"action_frequency": 30, "rendering_frequency": 30,
                "physics_frequency": 120, "automatic_reset": False,
                "flatten_obs_space": False, "device": None},
        "scene": {"type": "InteractiveTraversableScene",
                  "scene_model": "house_double_floor_lower",
                  "load_room_types": ["corridor", "living_room", "kitchen"],
                  "include_robots": False},
        "task": {"type": "BehaviorTask", "activity_name": "picking_up_trash",
                 "online_object_sampling": False, "include_obs": False,
                 "termination_config": {"max_steps": 50000},
                 "reward_config": {"r_potential": 1.0}},
        "objects": [],
        "robots": [_source_robot()],
    }
    cfg.update(overrides)
    return cfg


def _source_robot(**overrides) -> dict:
    robot = {
        "type": "R1Pro",
        "name": "robot_r1",
        "action_normalize": False,
        "controller_config": _controller_config(),
        "self_collisions": True,
        "obs_modalities": [],
        "position": [6.1107177734375, 1.5191650390625, 0.003753662109375],
        "orientation": [0.004324102774262428, -0.0026167959440499544,
                        -0.3354952037334442, 0.9420283436775208],
        "grasping_mode": "assisted",
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {
            "image_height": 1080, "image_width": 1080}}},
        "reset_joint_pos": _reset(),
    }
    robot.update(overrides)
    return robot


def test_build_ready_from_source():
    cfg = _source_cfg()
    built = CompatConfigBuilder(cfg, cfg["robots"][0]).build()
    assert built.ready, built.blocked
    assert built.env["action_frequency"] == 30
    assert built.env["rendering_frequency"] == 30
    assert built.env["physics_frequency"] == 120
    assert built.env["automatic_reset"] is False
    assert built.env["flatten_obs_space"] is True
    assert built.scene["scene_model"] == "house_double_floor_lower"
    assert built.task["activity_name"] == "picking_up_trash"
    assert built.robot["model"] == "r1pro"
    assert "type" not in built.robot
    assert built.robot["name"] == "robot_r1"
    assert built.robot["_source_uuid"] == 84011472
    assert list(built.robot["controller_config"].keys()) == list(TARGET_CONTROLLER_ORDER)
    assert len(built.robot["reset_joint_pos"]) == TARGET_RESET_COUNT
    assert built.robot["proprio_obs"] == list(TARGET_PROPRIO_KEYS)
    assert built.robot["obs_modalities"] == ["proprio", "rgb"]


def test_legacy_type_migrated_to_lowercase_model():
    robot = _source_robot()
    assert "model" not in robot and "type" in robot
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert built.ready, built.blocked
    assert built.robot["model"] == "r1pro"
    assert "type" not in built.robot


def test_frequency_mismatch_blocks():
    cfg = _source_cfg()
    cfg["env"]["physics_frequency"] = 60
    built = CompatConfigBuilder(cfg, cfg["robots"][0]).build()
    assert not built.ready
    assert any(b["reason"] == "frequency_mismatch" for b in built.blocked)


def test_scene_type_mismatch_blocks():
    cfg = _source_cfg()
    cfg["scene"]["type"] = "OtherScene"
    built = CompatConfigBuilder(cfg, cfg["robots"][0]).build()
    assert not built.ready
    assert any(b["reason"] == "scene_type_mismatch" for b in built.blocked)


def test_robot_model_mismatch_blocks():
    robot = _source_robot(type="NoSuchRobot")
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert not built.ready
    assert any(b["reason"] == "robot_model_mismatch" for b in built.blocked)


def test_unknown_controller_blocks():
    cc = _controller_config()
    cc["wrist_motor"] = {"name": "JointController"}
    robot = _source_robot(controller_config=cc)
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert not built.ready
    assert any(b["reason"] == "controller_unknown" for b in built.blocked)


def test_controller_set_missing_blocks():
    cc = _controller_config()
    del cc["trunk"]
    robot = _source_robot(controller_config=cc)
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert not built.ready
    assert any(b["reason"] == "controller_unknown" for b in built.blocked)


def test_controller_source_order_differs_does_not_block():
    cc = _controller_config()
    reordered = {"base": cc.pop("base"), **cc}
    robot = _source_robot(controller_config=reordered)
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert built.ready, built.blocked
    assert list(built.robot["controller_config"].keys()) == list(TARGET_CONTROLLER_ORDER)


def test_reset_count_mismatch_blocks():
    robot = _source_robot(reset_joint_pos=[0.0] * 23)
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert not built.ready
    assert any(b["reason"] == "reset_joint_count_mismatch" for b in built.blocked)


def test_reset_nonfinite_blocks():
    bad = _reset()
    bad[0] = float("nan")
    robot = _source_robot(reset_joint_pos=bad)
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert not built.ready
    assert any(b["reason"] == "reset_joint_nonfinite" for b in built.blocked)


def test_unsupported_robot_field_blocks():
    robot = _source_robot(custom_secret_field=1)
    built = CompatConfigBuilder(_source_cfg(), robot).build()
    assert not built.ready
    assert any(b["reason"] == "unsupported_robot_field" for b in built.blocked)


def test_effective_readback_matches():
    built = CompatConfigBuilder(_source_cfg(), _source_robot()).build()
    assert built.ready
    effective = {
        "env": {"action_frequency": 30, "rendering_frequency": 30,
                "physics_frequency": 120},
        "scene": {"type": "InteractiveTraversableScene",
                  "scene_model": built.scene["scene_model"]},
        "task": {"activity_name": built.task["activity_name"]},
        "robots": [{
            "model": "r1pro", "name": "robot_r1",
            "controller_config": {k: {} for k in TARGET_CONTROLLER_ORDER},
            "reset_joint_pos": [0.0] * TARGET_RESET_COUNT,
        }],
    }
    report = validate_effective(built, effective)
    assert report["ready"], report["mismatches"]
    assert not report["mismatches"]


def test_effective_readback_reports_mismatch():
    built = CompatConfigBuilder(_source_cfg(), _source_robot()).build()
    effective = {
        "env": {"action_frequency": 30, "rendering_frequency": 30,
                "physics_frequency": 60},
        "scene": {"type": "OtherScene", "scene_model": "wrong"},
        "task": {"activity_name": "other_task"},
        "robots": [{"type": "R1Pro", "model": "fetch", "name": "x",
                    "controller_config": {}, "reset_joint_pos": [0.0]}],
    }
    report = validate_effective(built, effective)
    assert not report["ready"]
    surfaces = {m["surface"] for m in report["mismatches"]}
    assert "env.physics_frequency" in surfaces
    assert "scene.type" in surfaces
    assert "robot.type" in surfaces
    assert "robot.model" in surfaces
    assert "robot.controller_config.order" in surfaces
