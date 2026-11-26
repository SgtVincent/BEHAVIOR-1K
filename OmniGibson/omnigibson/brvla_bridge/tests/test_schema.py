"""Canary schema: positive bundle passes; every contract breach rejects."""
import copy

from brvla_bridge import contracts, schema
from brvla_bridge.errors import SchemaContractError
from brvla_bridge.tests._assert import raises

EXPECTED_DATANAME = "ds-brvla45"  # public test placeholder; real id lives in private manifest


def _valid_obs():
    return {
        "dataname": EXPECTED_DATANAME,
        "instruction": "Task: turning_on_radio",
        "robot_type": "br2",
        "images": [
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]] for _ in range(3)
        ],
        "br_states": [[0.1] * contracts.STATE45 for _ in range(contracts.HISTORY_LEN)],
        "br_actions": [[0.0] * contracts.STATE45 for _ in range(contracts.CHUNK_SIZE)],
        "br_action_is_pad": [False] * contracts.CHUNK_SIZE,
    }


def test_valid_bundle_normalizes():
    out = schema.validate_canary_obs(_valid_obs(), expected_dataname=EXPECTED_DATANAME)
    assert out["robot_type"] == "br2"
    assert len(out["images"]) == 3
    assert len(out["br_states"]) == 4 and len(out["br_states"][0]) == 45
    assert len(out["br_actions"]) == 50


def test_rejects_wrong_robot_type():
    obs = _valid_obs(); obs["robot_type"] = "r1"
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_rejects_wrong_image_count():
    obs = _valid_obs(); obs["images"] = obs["images"][:2]
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_rejects_bad_state_shape():
    obs = _valid_obs(); obs["br_states"] = [[0.1] * 45 for _ in range(3)]
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_rejects_nonfinite_history():
    obs = _valid_obs()
    obs["br_states"][0][0] = float("nan")
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_rejects_nonzero_action_pad():
    obs = _valid_obs(); obs["br_actions"][0][0] = 1.0
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_rejects_bad_pad_mask():
    obs = _valid_obs(); obs["br_action_is_pad"][3] = True
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_rejects_bad_dataname():
    obs = _valid_obs(); obs["dataname"] = "other_run"
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_instruction_must_be_string():
    obs = _valid_obs(); obs["instruction"] = 42
    with raises(SchemaContractError):
        schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)


def test_obs_not_dict():
    with raises(SchemaContractError):
        schema.validate_canary_obs(["not", "a", "dict"])


def test_copy_does_not_mutate_input():
    obs = _valid_obs()
    before = copy.deepcopy(obs)
    schema.validate_canary_obs(obs, expected_dataname=EXPECTED_DATANAME)
    assert obs == before
