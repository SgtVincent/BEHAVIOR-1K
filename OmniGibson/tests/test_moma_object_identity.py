"""CPU tests for source-to-live object identity matching.

Sim-free: live objects are duck-typed fakes. The BDDL instance key binds a
role slot only; closure confirmation requires category/model/asset-hash and
source scene closure correspondence. Class-only alignment and role-key
coincidence never confirm the same initial state.
"""
import numpy as np

from omnigibson.moma.object_identity import (
    SourceObjectRef,
    SourceSceneRef,
    match_objects,
    parse_source_name,
    pose_residual,
    refs_from_anchor,
)


def _pose(translation=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    pose[:3, 3] = translation
    return pose


class _FakeLiveObj:
    def __init__(self, name, prim_path, category="object", translation=(0.0, 0.0, 0.0),
                 model="itolcg", in_rooms=None, expected_file_hash="b003114d"):
        self.name = name
        self.prim_path = prim_path
        self.relative_prim_path = prim_path
        self.category = category
        self.model = model
        self.in_rooms = in_rooms
        self._expected_file_hash = expected_file_hash
        self._usd_path = f"/fake/{model}.usd"
        self._uuid = f"uuid-{name}"
        self._translation = translation

    def get_position_orientation(self):
        return np.array(self._translation, dtype=float), np.array([0.0, 0.0, 0.0, 1.0])


def _scene():
    return SourceSceneRef(
        scene_model="house_double_floor_lower",
        scene_instance="house_double_floor_lower_task_picking_up_trash_0_0_template",
        config_attr_sha256="910cf79d",
        scene_file_attr_sha256="f1195d99",
        episode="episode_00010010.hdf5",
    )


def _live_closure():
    return {
        "scene_model": "house_double_floor_lower",
        "scene_instance": "house_double_floor_lower_task_picking_up_trash_0_0_template",
        "config_attr_sha256": "910cf79d",
        "scene_file_attr_sha256": "f1195d99",
    }


def _ref(name, translation=(0.0, 0.0, 0.0), **kwargs):
    kwargs.setdefault("scene", _scene())
    return SourceObjectRef(
        role=parse_source_name(name)["role"],
        source_name=name,
        source_pose=_pose(translation),
        source_index=818,
        **kwargs,
    )


def test_parse_source_name_splits_prefix_and_suffix():
    parsed = parse_source_name("can_of_soda_114")
    assert parsed["role"] == "can"
    assert parsed["prefix"] == "can_of_soda"
    assert parsed["suffix"] == "114"
    bucket = parse_source_name("trash_can_116")
    assert bucket["role"] == "bucket"
    assert bucket["suffix"] == "116"


def test_instance_binding_without_scene_closure_is_not_confirmed():
    live = _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2",
                        category="can_of_soda", model="itolcg",
                        in_rooms=["living_room_0"])
    ref = _ref("can_of_soda_114", bddl_instance="can__of__soda.n.01_2",
               category="can_of_soda", model="itolcg")
    report = match_objects([ref], [live], scope_keys=["can__of__soda.n.01_2"])
    match = report["matches"]["can_of_soda_114"]
    assert match["verdict"] == "role_bound_scene_unverified"
    assert match["identity_confirmed"] is False
    assert match["matched_live_name"] is None


def test_closure_confirmed_with_matching_scene_and_model():
    live = _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2",
                        category="can_of_soda", model="itolcg",
                        in_rooms=["living_room_0"])
    ref = _ref("can_of_soda_114", bddl_instance="can__of__soda.n.01_2",
               category="can_of_soda", model="itolcg",
               expected_file_hash="b003114d")
    report = match_objects([ref], [live], scope_keys=["can__of__soda.n.01_2"],
                           live_scene_closure=_live_closure())
    match = report["matches"]["can_of_soda_114"]
    assert match["verdict"] == "closure_confirmed"
    assert match["identity_confirmed"] is True
    assert match["matched_live_name"] == "can__of__soda.n.01_2"
    corr = match["candidates"][0]["corroboration"]
    assert corr["category_match"] is True
    assert corr["model_match"] is True
    assert corr["asset_hash_match"] is True
    assert report["identity_confirmed"] is True


def test_model_mismatch_is_rejected_not_confirmed():
    live = _FakeLiveObj("can__of__soda.n.01_3", "/World/can__of__soda.n.01_3",
                        category="can_of_soda", model="lugwcz")
    ref = _ref("can_of_soda_113", bddl_instance="can__of__soda.n.01_3",
               category="can_of_soda", model="itolcg")
    report = match_objects([ref], [live], scope_keys=["can__of__soda.n.01_3"],
                           live_scene_closure=_live_closure())
    match = report["matches"]["can_of_soda_113"]
    assert match["verdict"] == "rejected_model"
    assert match["identity_confirmed"] is False
    assert match["matched_live_name"] is None


def test_category_mismatch_is_rejected():
    live = _FakeLiveObj("can__of__soda.n.01_3", "/World/can__of__soda.n.01_3",
                        category="trash_can", model="lugwcz")
    ref = _ref("can_of_soda_113", bddl_instance="can__of__soda.n.01_3",
               category="can_of_soda", model="lugwcz")
    report = match_objects([ref], [live], scope_keys=["can__of__soda.n.01_3"],
                           live_scene_closure=_live_closure())
    match = report["matches"]["can_of_soda_113"]
    assert match["verdict"] == "rejected_category"
    assert match["identity_confirmed"] is False


def test_scene_closure_mismatch_is_rejected():
    live = _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2",
                        category="can_of_soda", model="itolcg")
    ref = _ref("can_of_soda_114", bddl_instance="can__of__soda.n.01_2",
               category="can_of_soda", model="itolcg")
    other_scene = dict(_live_closure(), scene_instance="some_other_template")
    report = match_objects([ref], [live], scope_keys=["can__of__soda.n.01_2"],
                           live_scene_closure=other_scene)
    match = report["matches"]["can_of_soda_114"]
    assert match["verdict"] == "rejected_scene_closure"
    assert match["identity_confirmed"] is False


def test_asset_hash_mismatch_is_rejected():
    live = _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2",
                        category="can_of_soda", model="itolcg",
                        expected_file_hash="deadbeef")
    ref = _ref("can_of_soda_114", bddl_instance="can__of__soda.n.01_2",
               category="can_of_soda", model="itolcg",
               expected_file_hash="b003114d")
    report = match_objects([ref], [live], scope_keys=["can__of__soda.n.01_2"],
                           live_scene_closure=_live_closure())
    match = report["matches"]["can_of_soda_114"]
    assert match["verdict"] == "rejected_asset_hash"
    assert match["identity_confirmed"] is False


def test_class_only_match_is_unconfirmed():
    live = _FakeLiveObj("can__of__soda.n.01_1", "/World/can__of__soda.n.01_1",
                        category="can_of_soda", translation=(0.1, 0.2, 0.3))
    ref = _ref("can_of_soda_115", translation=(0.1, 0.2, 0.3))
    report = match_objects([ref], [live])
    match = report["matches"]["can_of_soda_115"]
    assert match["verdict"] == "unconfirmed_role_pose"
    assert match["identity_confirmed"] is False
    assert match["matched_live_name"] is None


def test_role_only_match_within_pose_still_unconfirmed():
    live = _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2",
                        category="can_of_soda")
    report = match_objects([_ref("can_of_soda_114", translation=(9.0, 9.0, 9.0))], [live])
    match = report["matches"]["can_of_soda_114"]
    assert match["verdict"] == "unconfirmed_role_only"
    assert match["identity_confirmed"] is False


def test_two_role_candidates_are_ambiguous_unconfirmed():
    lives = [
        _FakeLiveObj("can__of__soda.n.01_1", "/World/can__of__soda.n.01_1",
                     translation=(0.1, 0.2, 0.3)),
        _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2",
                     translation=(0.1, 0.2, 0.3)),
    ]
    report = match_objects([_ref("can_of_soda_114", translation=(0.1, 0.2, 0.3))], lives)
    match = report["matches"]["can_of_soda_114"]
    assert match["role_candidate_count"] == 2
    assert match["verdict"] == "unconfirmed_role_only"
    assert match["identity_confirmed"] is False


def test_no_role_match_refuses():
    live = _FakeLiveObj("ashcan.n.01_1", "/World/ashcan.n.01_1", category="ashcan")
    report = match_objects([_ref("can_of_soda_113")], [live])
    match = report["matches"]["can_of_soda_113"]
    assert match["verdict"] == "no_role_match"
    assert match["identity_confirmed"] is False


def test_source_identity_fields_default_unknown():
    ref = _ref("trash_can_116")
    record = ref.record()
    assert record["source_uid"] == "UNKNOWN"
    assert record["category"] == "UNKNOWN"
    assert record["model"] == "UNKNOWN"
    assert record["bddl_instance"] == "UNKNOWN"


def test_refs_from_anchor_uses_single_epoch_se3_no_blend():
    matrix = [
        [0.99, 0.12, 0.0, 4.64],
        [-0.12, 0.99, 0.0, 6.48],
        [0.0, 0.0, 1.0, 0.065],
        [0.0, 0.0, 0.0, 1.0],
    ]
    anchor_rows = [{
        "source_name": "can_of_soda_113",
        "bddl_instance": "can__of__soda.n.01_3",
        "category": "can_of_soda",
        "model": "lugwcz",
        "expected_file_hash": "da746049",
        "in_rooms": ["living_room_0"],
        "registry_root_link_pose": {"pos": [4.24, 6.99, 0.065], "ori": [0, 0, 0.587, 0.81],
                                     "epoch": "scene_registry_initial"},
        "npz_pose_frame0": matrix,
        "npz_pose_frame818": matrix,
    }]
    refs = refs_from_anchor(anchor_rows, scene=_scene())
    ref = refs[0]
    assert ref.bddl_instance == "can__of__soda.n.01_3"
    assert ref.model == "lugwcz"
    np.testing.assert_allclose(np.asarray(ref.source_pose), np.asarray(matrix))
    assert ref.extras["registry_root_link_pose"]["epoch"] == "scene_registry_initial"
    assert ref.extras["pose_epoch"] == "npz_pose_frame818"


def test_refs_from_anchor_incomplete_pose_marks_unknown_no_blend():
    anchor_rows = [{
        "source_name": "trash_can_116",
        "bddl_instance": "ashcan.n.01_1",
        "category": "trash_can",
        "model": "wkxtxh",
        "expected_file_hash": "3a2fb273",
        "in_rooms": ["kitchen_0"],
        "registry_root_link_pose": {"pos": [6.51, 1.23, 0.135], "ori": [0, 0, 0.932, 0.361]},
        "npz_pose_frame0": [6.89, 2.75, 0.135],
        "npz_pose_frame818": [6.89, 2.75, 0.135],
    }]
    refs = refs_from_anchor(anchor_rows)
    ref = refs[0]
    assert isinstance(ref.source_pose, dict)
    assert ref.source_pose["status"] == "UNKNOWN"
    assert "no_cross_epoch_blend" in ref.source_pose["error"]


def test_pose_residual_reports_translation_and_rotation():
    a = _pose((0.0, 0.0, 0.0))
    b = _pose((0.1, 0.0, 0.0))
    residual = pose_residual(a, b)
    assert abs(residual["translation_m"] - 0.1) < 1e-9
    assert residual["rotation_deg"] < 1e-9
    assert residual["finite"]
