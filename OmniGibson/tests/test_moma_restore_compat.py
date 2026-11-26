"""CPU tests for the source-compatible restore bootstrap / reader contract.

Sim-free: live objects are duck-typed fakes. The post1 uuid formula is
replicated from the original source and checked against the values captured
from the post1 checkout (probe 124). Any unmapped required segment blocks the
entire restore; a partial state set is never a success.
"""
import numpy as np

from omnigibson.moma.restore_compat import (
    BOOTSTRAP_STEPS,
    FRAME_CHECKPOINTS,
    PARTIAL_SUCCESS_ALLOWED,
    SourceCompatReader,
    SourcePin,
    TargetMapEntry,
    compute_uuid,
)

# Values captured read-only from post1 `get_uuid` (probe 124, trial 458773744).
POST1_UUID_VECTOR = {
    "robot_r1": 84011472,
    "trash_can_116": 81627904,
    "can_of_soda_115": 91772192,
    "can_of_soda_114": 13654295,
    "can_of_soda_113": 43525804,
    "floors_ulujpr_0": 54272132,
    "floors_zqjkvm_0": 88837304,
}


class _FakeLiveObj:
    def __init__(self, name, category="can_of_soda", model="itolcg",
                 in_rooms=None, expected_file_hash="b003114df6a0addaa3a87687339d8935",
                 translation=(0.0, 0.0, 0.0)):
        self.name = name
        self.prim_path = f"/World/{name}"
        self.relative_prim_path = self.prim_path
        self.category = category
        self.model = model
        self.in_rooms = in_rooms or ["living_room_0"]
        self._expected_file_hash = expected_file_hash
        self._usd_path = f"/fake/{model}.usd"
        self._uuid = f"uuid-{name}"
        self._translation = translation

    def get_position_orientation(self):
        return np.array(self._translation, dtype=float), np.array(
            [0.0, 0.0, 0.0, 1.0])


def _pin():
    return SourcePin(
        demo_sha256="66c8396817198a2a165225adae56de748cc0598f16db0edf073e7a47760bf45b",
        config_attr_sha256="910cf79d78d8f8e8464559ceab06fa2651fe847afd49d78fa3079b03711140c5",
        scene_file_attr_sha256="f1195d994f29ff2661c46735a4bfae208d37fbaeee5e22fe93fd224b0bcd1893",
        episode="episode_00010010.hdf5",
        scene_model="house_double_floor_lower",
        scene_instance="house_double_floor_lower_task_picking_up_trash_0_0_template",
        state_schema={"state_dtype": "float32", "state_size_818": 150},
    )


def _live_closure():
    return {
        "scene_model": "house_double_floor_lower",
        "scene_instance": "house_double_floor_lower_task_picking_up_trash_0_0_template",
        "config_attr_sha256": "910cf79d78d8f8e8464559ceab06fa2651fe847afd49d78fa3079b03711140c5",
        "scene_file_attr_sha256": "f1195d994f29ff2661c46735a4bfae208d37fbaeee5e22fe93fd224b0bcd1893",
    }


def _source_attrs():
    return {
        "demo_sha256": "66c8396817198a2a165225adae56de748cc0598f16db0edf073e7a47760bf45b",
        "config_attr_sha256": "910cf79d78d8f8e8464559ceab06fa2651fe847afd49d78fa3079b03711140c5",
        "scene_file_attr_sha256": "f1195d994f29ff2661c46735a4bfae208d37fbaeee5e22fe93fd224b0bcd1893",
        "episode": "episode_00010010.hdf5",
    }


def _can_entry(**kwargs):
    defaults = dict(
        source_name="can_of_soda_115",
        bddl_instance="can__of__soda.n.01_1",
        category="can_of_soda",
        model="itolcg",
        expected_file_hash="b003114df6a0addaa3a87687339d8935",
        in_rooms=["living_room_0"],
        uuid_mode="name_preserved",
    )
    defaults.update(kwargs)
    return TargetMapEntry(**defaults)


def test_uuid_formula_matches_post1_vector():
    for name, expected in POST1_UUID_VECTOR.items():
        assert compute_uuid(name) == expected
        assert compute_uuid(name) == compute_uuid(name)  # deterministic


def test_formula_rejects_empty_name():
    try:
        compute_uuid("")
    except ValueError as exc:
        assert "uuid_requires_nonempty_name" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty name")


def test_name_preserved_closure_confirmed_is_ready():
    entry = _can_entry()
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("can_of_soda_115")]
    plan = reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=_live_closure(), source_attrs=_source_attrs())
    assert plan.ready is True
    assert plan.status == "ready"
    assert len(plan.blocked) == 0
    mapped = plan.mapped[0]
    assert mapped["verdict"] == "closure_confirmed"
    assert mapped["uuid_alignment"]["aligned"] is True
    assert mapped["uuid_alignment"]["reason"] == "name_preserved_uuid_formula_match"


def test_role_only_without_scene_closure_blocks():
    entry = _can_entry()
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("can_of_soda_115")]
    plan = reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=None, source_attrs=_source_attrs())
    assert plan.ready is False
    assert plan.status == "restore_blocked"
    blocked = plan.blocked[0]
    assert "scene_closure_unverified" in blocked["reasons"]


def test_model_mismatch_is_rejected_and_blocks():
    entry = _can_entry(model="lugwcz")
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("can_of_soda_115", model="itolcg")]
    plan = reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=_live_closure(), source_attrs=_source_attrs())
    assert plan.status == "restore_blocked"
    blocked = plan.blocked[0]
    assert blocked["verdict"] == "rejected_model"
    assert "closure_rejected" in blocked["reasons"]


def test_renamed_target_without_remap_blocks():
    entry = _can_entry()
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("soda_can_target_7")]
    plan = reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=_live_closure(), source_attrs=_source_attrs())
    assert plan.status == "restore_blocked"
    blocked = plan.blocked[0]
    assert "uuid_unaligned_name_renamed" in blocked["reasons"]


def test_explicit_remap_evidenced_is_ready():
    target_uuid = compute_uuid("soda_can_target_7")
    entry = _can_entry(
        target_name="soda_can_target_7",
        uuid_mode="explicit_remap",
        explicit_target_uuid=target_uuid)
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("soda_can_target_7")]
    plan = reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=_live_closure(), source_attrs=_source_attrs())
    assert plan.ready is True
    aligned = plan.mapped[0]["uuid_alignment"]
    assert aligned["aligned"] is True
    assert aligned["reason"] == "explicit_remap_evidenced"


def test_explicit_remap_without_target_uuid_blocks():
    entry = _can_entry(
        target_name="soda_can_target_7",
        uuid_mode="explicit_remap",
        explicit_target_uuid=None)
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("soda_can_target_7")]
    plan = reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=_live_closure(), source_attrs=_source_attrs())
    assert plan.status == "restore_blocked"
    blocked = plan.blocked[0]
    assert "required_segment_unmapped" in blocked["reasons"]


def test_missing_required_object_blocks_entire_restore():
    entry = _can_entry()
    reader = SourceCompatReader(_pin(), [entry])
    plan = reader.prepare_restore(
        [], scope_keys=[], live_scene_closure=_live_closure(),
        source_attrs=_source_attrs())
    assert plan.status == "restore_blocked"
    assert PARTIAL_SUCCESS_ALLOWED is False
    assert plan.blocked[0]["reasons"]


def test_source_pin_mismatch_blocks():
    entry = _can_entry()
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("can_of_soda_115")]
    bad_attrs = dict(_source_attrs())
    bad_attrs["demo_sha256"] = "0" * 64
    plan = reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=_live_closure(), source_attrs=bad_attrs)
    assert plan.status == "restore_blocked"
    assert any(b["reason"] == "source_pin_mismatch" for b in plan.blocked)


def test_reader_does_not_mutate_source_group():
    entry = _can_entry()
    reader = SourceCompatReader(_pin(), [entry])
    live = [_FakeLiveObj("can_of_soda_115")]
    reader.prepare_restore(
        live, scope_keys=["can__of__soda.n.01_1"],
        live_scene_closure=_live_closure(), source_attrs=_source_attrs())
    assert reader._source_group_was_written is False


def test_bootstrap_order_and_checkpoints_are_frozen():
    assert BOOTSTRAP_STEPS[0] == "create_env_read_attrs_config_scene_file"
    assert BOOTSTRAP_STEPS[-1] == "pose_invalidate_render"
    assert FRAME_CHECKPOINTS == ("frame0", "grasp", "physics30", "physics120")
    assert "replay_state_0_to_818_serialized" in BOOTSTRAP_STEPS
