"""CPU tests for source-to-live object identity matching.

Sim-free: live objects are duck-typed fakes. Class-only alignment must never
be accepted as the same initial state; only an exact name/prim-path match
confirms identity.
"""
import numpy as np

from omnigibson.moma.object_identity import (
    SourceObjectRef,
    match_objects,
    parse_source_name,
    pose_residual,
)


def _pose(translation=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    pose[:3, 3] = translation
    return pose


class _FakeLiveObj:
    def __init__(self, name, prim_path, category="object", translation=(0.0, 0.0, 0.0),
                 model="/fake/asset.usd"):
        self.name = name
        self.prim_path = prim_path
        self.relative_prim_path = prim_path
        self.category = category
        self._usd_path = model
        self._uuid = f"uuid-{name}"
        self._translation = translation

    def get_position_orientation(self):
        return np.array(self._translation, dtype=float), np.array([0.0, 0.0, 0.0, 1.0])


def _ref(name, translation=(0.0, 0.0, 0.0)):
    return SourceObjectRef(
        role=parse_source_name(name)["role"],
        source_name=name,
        source_pose=_pose(translation),
        source_index=818,
    )


def test_parse_source_name_splits_prefix_and_suffix():
    parsed = parse_source_name("can_of_soda_114")
    assert parsed["role"] == "can"
    assert parsed["prefix"] == "can_of_soda"
    assert parsed["suffix"] == "114"
    bucket = parse_source_name("trash_can_116")
    assert bucket["role"] == "bucket"
    assert bucket["suffix"] == "116"


def test_name_exact_match_confirms_identity():
    live = _FakeLiveObj("can_of_soda_114", "/World/can_of_soda_114", category="soda_can")
    report = match_objects([_ref("can_of_soda_114")], [live])
    match = report["matches"]["can_of_soda_114"]
    assert match["verdict"] == "name_exact"
    assert match["identity_confirmed"] is True
    assert match["matched_live_name"] == "can_of_soda_114"
    assert report["identity_confirmed"] is True


def test_class_only_match_is_unconfirmed():
    live = _FakeLiveObj("can__of__soda.n.01_1", "/World/can__of__soda.n.01_1",
                        category="can__of__soda.n.01", translation=(0.1, 0.2, 0.3))
    ref = _ref("can_of_soda_115", translation=(0.1, 0.2, 0.3))
    report = match_objects([ref], [live])
    match = report["matches"]["can_of_soda_115"]
    assert match["verdict"] == "unconfirmed_role_pose"
    assert match["identity_confirmed"] is False
    assert match["matched_live_name"] is None


def test_role_only_match_within_pose_still_unconfirmed():
    live = _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2",
                        category="can__of__soda.n.01")
    report = match_objects([_ref("can_of_soda_114", translation=(9.0, 9.0, 9.0))], [live])
    match = report["matches"]["can_of_soda_114"]
    assert match["verdict"] == "unconfirmed_role_only"
    assert match["identity_confirmed"] is False


def test_two_role_candidates_are_ambiguous_unconfirmed():
    lives = [
        _FakeLiveObj("can__of__soda.n.01_1", "/World/can__of__soda.n.01_1", translation=(0.1, 0.2, 0.3)),
        _FakeLiveObj("can__of__soda.n.01_2", "/World/can__of__soda.n.01_2", translation=(0.1, 0.2, 0.3)),
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


def test_pose_residual_reports_translation_and_rotation():
    a = _pose((0.0, 0.0, 0.0))
    b = _pose((0.1, 0.0, 0.0))
    residual = pose_residual(a, b)
    assert abs(residual["translation_m"] - 0.1) < 1e-9
    assert residual["rotation_deg"] < 1e-9
    assert residual["finite"]
