"""Stdlib tests for the model physical23 chunk decoder seam.

Covers cumulative-profile decompose against the fixed frame0 anchor, source
checkpoint receipt binding, current-q inactive-arm hold, model-sourced base /
gripper groups, scope block, and the policy native23 composition path. The
fixture reuses one real cached row captured offline in attempt3; the full
50x23 chunk was not dumped, so ``chunk_full_recomputed`` is asserted False and
no full-chunk hash is fabricated.

No planner, simulator, torch, numpy, or omnigibson is imported.
"""
import math

from brvla_bridge import contracts, delta_tf, managed_policy, model_decoder, native_action, scope
from brvla_bridge.errors import BackendNotDeclaredError, ContractViolation

DT = 1.0 / 30.0

# Real cached chunk row captured offline in attempt3 (step3, 23-wide finite).
REAL_ROW = [
    -0.006301412869928521, -0.004298036852360238, -0.001310242457024463,
    0.003693274983171374, -0.02341950399868563, -0.010508602160679642,
    -0.017838139186946676, 0.000452151236021308, 0.0025132578184902665,
    -0.0013748659501685717, -0.030222908201607876, 0.005746569530580781,
    0.0488281738455703, 0.052734433857778484,
    0.0019531709431073433, -0.00022701478532294317, 0.0005374733914970398,
    -0.00013321373645440275, -7.2268170573886945e-06,
    -3.221990518068041e-07, 7.941502318267041e-05,
    -0.00011374787479260379, 0.00021089032719864148,
]

SOURCE_RECEIPT = {
    "checkpoint_index_sha256": "b49e94981bf8a2cbd2d72a6c54ded3e1d366734ff6947eb8945b2a7a0f83f7f0",
    "stats_sha256": "6029a10e04811b14f0c2fef527bb14b23a41d6e30bb272866f838f12519bff3c",
    "server_sha256": "314dd555b56c620f2f6e9f24d3853f65ba235b438e136a3b2f4e54ea6329ab8a",
}


def _anchors() -> dict:
    return {
        "left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "right": [-0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        "chest": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
    }


def _receipt(**overrides):
    certificate = {
        "status": "optimal", "bound_arm": "left", "primary_tool": "left_gripper_link",
        "certified_scope": "left_arm_only; self_collision; scene_model_null",
        "right_arm_planned": False, "device_uuid": "GPU-abc",
        "device_uuid_source": "manifest_argument_not_live_probe",
        "receipt": "r", "config_yml_sha256": "c",
        "scene_identity": None, "source_epoch": None,
    }
    certificate.update(overrides)
    return {
        "gate_pass": True,
        "stages": {
            "runtime": {"status": "ok"}, "bind": {"status": "ok"},
            "backend_plan": {"status": "ok", "outputs": {"success": True, "certificate": certificate}},
            "executor_plan": {"status": "ok", "outputs": {
                "vector_width": 23, "vector_finite": True, "certificate": certificate,
                "meta": {"waypoints": 3, "batch": 1, "horizon": 1, "dt": 0.025},
            }},
        },
    }


def _scope(**overrides) -> scope.ScopeReport:
    return scope.verify_receipt(_receipt(**overrides), required_arm="left")


class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.waypoints = [[float(k + 1)] * contracts.ACTION23 for k in range(3)]

    def plan(self, goal):
        self.calls.append(goal)
        return {"action23": None, "certificate": {}, "trajectory": {
            "native23_waypoints": self.waypoints, "dt": 0.025,
            "batch": 1, "horizon": 1, "waypoints": len(self.waypoints),
        }}


def _chunk() -> list:
    row1 = [0.0] * contracts.ACTION23
    row1[12:14] = [0.3, 0.5]
    row1[14:17] = [0.0, 0.0, 0.0]
    return [list(REAL_ROW), row1]


def _decoder(**kwargs) -> model_decoder.ModelChunkDecoder:
    defaults = dict(
        chunk=_chunk(), anchors=_anchors(), dt_seconds=DT,
        scope_report=_scope(), executor=FakeExecutor(), source_receipt=SOURCE_RECEIPT,
    )
    defaults.update(kwargs)
    return model_decoder.ModelChunkDecoder(**defaults)


def test_decoder_receipt_binds_source_and_cumulative_profile():
    decoder = _decoder()
    receipt = decoder.receipt()
    assert receipt["representation"] == delta_tf.REPRESENTATION_CUMULATIVE
    assert receipt["model_layout"] == contracts.SERVER_METADATA_LAYOUT
    assert receipt["rows"] == 2
    # Only a partial offline capture exists; no full-chunk hash is fabricated.
    assert receipt["chunk_full_recomputed"] is False
    assert len(receipt["row_sha256"]) == 2
    for field, value in SOURCE_RECEIPT.items():
        assert receipt["source"][field] == value
    assert receipt["anchors"]["left"] == _anchors()["left"]


def test_decoder_rejects_missing_source_field():
    bad = dict(SOURCE_RECEIPT)
    del bad["stats_sha256"]
    try:
        _decoder(source_receipt=bad)
    except model_decoder.ModelDecoderViolation as exc:
        assert "stats_sha256" in str(exc)
    else:
        raise AssertionError("missing source field must fail closed")


def test_decoder_rejects_wrong_row_width():
    bad = [row[:-1] for row in _chunk()]
    try:
        _decoder(chunk=bad)
    except model_decoder.ModelDecoderViolation:
        return
    raise AssertionError("wrong row width must fail closed")


def test_decoder_rejects_nonfinite_row():
    bad = [list(REAL_ROW), [float("inf")] * contracts.ACTION23]
    try:
        _decoder(chunk=bad)
    except model_decoder.ModelDecoderViolation:
        return
    raise AssertionError("non-finite row must fail closed")


def test_decoder_rejects_empty_and_oversized_chunk():
    try:
        _decoder(chunk=[])
    except model_decoder.ModelDecoderViolation:
        pass
    else:
        raise AssertionError("empty chunk must fail closed")
    try:
        _decoder(chunk=[[0.0] * contracts.ACTION23] * (contracts.CHUNK_SIZE + 1))
    except model_decoder.ModelDecoderViolation:
        return
    raise AssertionError("oversized chunk must fail closed")


def test_decoder_rejects_bad_dt():
    try:
        _decoder(dt_seconds=0.0)
    except model_decoder.ModelDecoderViolation:
        return
    raise AssertionError("non-positive dt must fail closed")


def test_failed_scope_blocks_decoder_construction():
    failed = scope.verify_receipt(_receipt(), required_arm="right")
    try:
        _decoder(scope_report=failed)
    except ContractViolation:
        return
    raise AssertionError("failed scope must block decoder construction")


def test_target_pose_anchor_pose6_row_without_pairwise_accumulation():
    decoder = _decoder()
    anchors = _anchors()
    for index, row in enumerate(_chunk()):
        target = decoder.targets_at(index)
        for limb, sl in (("left", slice(0, 6)), ("right", slice(6, 12)),
                          ("chest", slice(17, 23))):
            expected = delta_tf.mat_mul(
                delta_tf.pose6_to_matrix(anchors[limb]),
                delta_tf.pose6_to_matrix(row[sl]),
            )
            actual = target["pose_matrices"][limb]
            for r in range(4):
                for c in range(4):
                    assert math.isclose(actual[r][c], expected[r][c], abs_tol=1e-12)
    # Cumulative profile must not carry row0's pose into row1: row1 pose is the
    # anchor applied to row1 alone, not an accumulation over row0.
    row1 = _chunk()[1]
    expected_left = delta_tf.mat_mul(
        delta_tf.pose6_to_matrix(anchors["left"]),
        delta_tf.pose6_to_matrix(row1[0:6]),
    )
    actual_left = decoder.targets_at(1)["pose_matrices"]["left"]
    for r in range(4):
        for c in range(4):
            assert math.isclose(actual_left[r][c], expected_left[r][c], abs_tol=1e-12)


def test_base_and_gripper_come_from_model_target():
    decoder = _decoder()
    target = decoder.targets_at(0)
    assert target["continuous_gripper"] == [REAL_ROW[12], REAL_ROW[13]]
    assert len(target["base_velocity_body"]) == 3
    assert all(math.isfinite(v) for v in target["base_velocity_body"])


def test_policy_assembles_native23_with_held_right_current_q():
    policy = managed_policy.ModelChunkPolicy(
        chunk=_chunk(), anchors=_anchors(), dt_seconds=DT, receipt=_receipt(),
        executor=FakeExecutor(), source_receipt=SOURCE_RECEIPT,
    )
    held = [0.5] * 7
    policy._decoder.bind_current_q("right", held, source_sha256="sha-current-q")
    command = policy._next_native23()
    assert len(command["vector"]) == contracts.ACTION23
    assert all(math.isfinite(v) for v in command["vector"])
    assert command["sources"]["arm_right"] == native_action.SOURCE_CURRENT_Q
    assert command["vector"][15:22] == held
    assert command["held_current_q"]["arm_right"]["source_sha256"] == "sha-current-q"
    # Replay-sourced groups come from the model target seam, not planner zero.
    assert command["sources"]["gripper_left"] == "replay"
    assert command["sources"]["gripper_right"] == "replay"
    assert command["sources"]["base"] == "replay"
    assert command["vector"][14] == REAL_ROW[12]
    assert command["vector"][22] == REAL_ROW[13]


def test_right_inactive_unbound_fails_closed():
    policy = managed_policy.ModelChunkPolicy(
        chunk=_chunk(), anchors=_anchors(), dt_seconds=DT, receipt=_receipt(),
        executor=FakeExecutor(), source_receipt=SOURCE_RECEIPT,
    )
    try:
        policy._next_native23()
    except ContractViolation as exc:
        assert "arm_right" in str(exc)
    else:
        raise AssertionError("unbound inactive right arm must fail closed")


def test_missing_executor_fails_closed():
    decoder = _decoder(executor=None)
    try:
        decoder.plan_row(0)
    except BackendNotDeclaredError:
        return
    raise AssertionError("missing executor must raise BackendNotDeclaredError")
