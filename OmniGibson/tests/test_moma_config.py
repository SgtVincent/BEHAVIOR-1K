"""CPU tests for the frozen a28 effective runner configuration."""
from omnigibson.moma.config import (
    RunnerConfig,
    FROZEN_CAN_NAMES,
    FROZEN_BUCKET_NAME,
    FROZEN_TASK_NAME,
)


def _cfg(**overrides):
    return RunnerConfig(run_id="trash_a28_cpu_test", **overrides)


def test_frozen_baseline_defaults_preserved():
    cfg = _cfg()
    assert cfg.task_name == FROZEN_TASK_NAME
    assert cfg.can_names == FROZEN_CAN_NAMES
    assert cfg.bucket_name == FROZEN_BUCKET_NAME
    assert cfg.step_hz == 30 and cfg.physics_hz == 120
    assert cfg.policy_mode == "external_s2"
    assert cfg.assisted_grasping is True
    assert cfg.validate() == []


def test_s2_carries_contract_view():
    cfg = _cfg()
    view = cfg.s2_cfg()
    assert view["can_names"] == FROZEN_CAN_NAMES
    assert view["bucket_name"] == FROZEN_BUCKET_NAME
    assert set(view["allowed_primitives"]) == set(cfg.allowed_primitives)
    assert view["primitive_caps"] == cfg.primitive_caps


def test_digest_stable_and_changes_with_config():
    cfg_a = _cfg()
    cfg_b = _cfg()
    assert cfg_a.effective_digest() == cfg_b.effective_digest()
    cfg_c = _cfg(max_steps=7001)
    assert cfg_a.effective_digest() != cfg_c.effective_digest()


def test_invalid_run_id_rejected():
    cfg = RunnerConfig(run_id="../escape")
    problems = cfg.validate()
    assert "invalid_run_id" in problems


def test_bucket_also_can_rejected():
    cfg = _cfg(can_names=[FROZEN_BUCKET_NAME], bucket_name=FROZEN_BUCKET_NAME)
    assert "bucket_is_also_can" in cfg.validate()


def test_cap_without_primitive_rejected():
    cfg = _cfg(primitive_caps={"unknown_primitive": 10})
    problems = cfg.validate()
    assert any(p.startswith("cap_without_primitive") for p in problems)


def test_unknown_primitive_rejected():
    cfg = _cfg(allowed_primitives=["not_a_real_primitive"])
    problems = cfg.validate()
    assert any(p.startswith("unknown_primitives") for p in problems)
