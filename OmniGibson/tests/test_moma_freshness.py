"""CPU unit tests for the snapshot freshness / decision-identity gate.

Ported rejection codes mirror the a28 ``submit_primitive_request`` boundary.
Sim-free: no omnigibson, torch, simulator or executor import.
"""
import pytest

from omnigibson.moma.freshness import (
    DecisionFreshnessRejected,
    binding_from_envelope,
    build_envelope,
    validate_binding,
)
from omnigibson.moma.config import RunnerConfig
from omnigibson.moma.operator import EpisodeConfig

RUN_ID = "run-freshness-a28"


def _envelope(*, version=5, entry=1, sim=10.0, entry_sim=0.0, obs="obs_abc_d0001"):
    env = build_envelope(
        run_id=RUN_ID,
        observation_id=obs,
        snapshot_version=version,
        decision_entry_snapshot_version=entry,
        decision_seq=1,
        sim_time_s=sim,
        max_age_steps=1800,
        max_age_sim_s=60.0,
    )
    env["decision_entry_sim_time_s"] = entry_sim
    return env


def _validate(env, binding, historical=None):
    validate_binding(
        binding=binding,
        envelope=env,
        historical_observation_ids=set(historical or []),
        run_id=RUN_ID,
        max_age_steps=1800,
        max_age_sim_s=60.0,
    )


def test_valid_current_binding_passes():
    env = _envelope()
    binding = binding_from_envelope(env)
    _validate(env, binding)


def test_binding_missing_is_rejected():
    env = _envelope()
    with pytest.raises(DecisionFreshnessRejected, match="binding_not_dict"):
        _validate(env, None)  # type: ignore[arg-type]


def test_run_id_mismatch_rejected():
    env = _envelope()
    binding = binding_from_envelope(env)
    binding["run_id"] = "other-run"
    with pytest.raises(DecisionFreshnessRejected, match="run_id_mismatch"):
        _validate(env, binding)


def test_missing_observation_id_rejected():
    env = _envelope()
    binding = binding_from_envelope(env)
    binding["observation_id"] = None
    with pytest.raises(DecisionFreshnessRejected, match="missing_observation_id"):
        _validate(env, binding)


def test_historical_observation_id_is_stale():
    env = _envelope(obs="obs_abc_d0002")
    old = "obs_abc_d0001"
    binding = binding_from_envelope(env)
    binding["observation_id"] = old
    with pytest.raises(DecisionFreshnessRejected, match="stale_observation_id"):
        _validate(env, binding, historical=[old])


def test_unrecognized_observation_id_is_invalid():
    env = _envelope()
    binding = binding_from_envelope(env)
    binding["observation_id"] = "obs_unknown_d0009"
    with pytest.raises(DecisionFreshnessRejected, match="invalid_observation_id"):
        _validate(env, binding)


def test_bool_snapshot_version_rejected():
    env = _envelope()
    binding = binding_from_envelope(env)
    binding["snapshot_version"] = True
    with pytest.raises(DecisionFreshnessRejected, match="invalid_snapshot_version"):
        _validate(env, binding)


def test_snapshot_version_below_entry_rejected():
    env = _envelope(version=10, entry=5)
    binding = binding_from_envelope(env)
    binding["snapshot_version"] = 4
    with pytest.raises(DecisionFreshnessRejected, match="stale_or_invalid_snapshot_version"):
        _validate(env, binding)


def test_snapshot_version_above_current_rejected():
    env = _envelope(version=10, entry=5)
    binding = binding_from_envelope(env)
    binding["snapshot_version"] = 11
    with pytest.raises(DecisionFreshnessRejected, match="stale_or_invalid_snapshot_version"):
        _validate(env, binding)


def test_snapshot_age_steps_at_limit_still_passes():
    env = _envelope(version=1801, entry=1)
    binding = binding_from_envelope(env)
    binding["snapshot_version"] = 1
    _validate(env, binding)


def test_snapshot_age_steps_expired_rejected():
    env = _envelope(version=1802, entry=1)
    binding = binding_from_envelope(env)
    binding["snapshot_version"] = 1
    with pytest.raises(DecisionFreshnessRejected, match="stale_observation_snapshot_expired"):
        _validate(env, binding)


def test_bool_sim_time_rejected():
    env = _envelope()
    binding = binding_from_envelope(env)
    binding["observed_sim_time_s"] = False
    with pytest.raises(DecisionFreshnessRejected, match="invalid_observed_sim_time_s"):
        _validate(env, binding)


def test_sim_time_before_entry_rejected():
    env = _envelope(sim=10.0, entry_sim=5.0)
    binding = binding_from_envelope(env)
    binding["observed_sim_time_s"] = 4.0
    with pytest.raises(DecisionFreshnessRejected, match="stale_or_invalid_snapshot_sim_time"):
        _validate(env, binding)


def test_sim_time_after_current_rejected():
    env = _envelope(sim=10.0, entry_sim=0.0)
    binding = binding_from_envelope(env)
    binding["observed_sim_time_s"] = 11.0
    with pytest.raises(DecisionFreshnessRejected, match="stale_or_invalid_snapshot_sim_time"):
        _validate(env, binding)


def test_sim_time_age_expired_rejected():
    env = _envelope(sim=61.0, entry_sim=0.0)
    binding = binding_from_envelope(env)
    binding["observed_sim_time_s"] = 0.0
    with pytest.raises(DecisionFreshnessRejected, match="stale_observation_snapshot_expired"):
        _validate(env, binding)


def test_missing_snapshot_version_allowed_within_wait_window():
    env = _envelope()
    binding = binding_from_envelope(env)
    binding["snapshot_version"] = None
    binding["observed_sim_time_s"] = None
    _validate(env, binding)


def test_frozen_freshness_constants_unchanged():
    runner = RunnerConfig(run_id=RUN_ID)
    assert runner.snapshot_max_age_steps == 1800
    assert runner.snapshot_max_age_sim_s == 60.0
    episode = EpisodeConfig(
        run_id=RUN_ID, can_names=[], bucket_name="trash_can_116", max_steps=10
    )
    assert episode.max_snapshot_age_steps == 1800
    assert episode.max_snapshot_age_sim_s == 60.0
