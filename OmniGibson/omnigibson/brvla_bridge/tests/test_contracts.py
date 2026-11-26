"""Contract constants: widths, layouts, gates, 2026 entry, wire keys."""
from brvla_bridge import contracts


def test_widths_are_frozen():
    assert contracts.CONTROL_HZ == 30
    assert contracts.HORIZON == 32
    assert contracts.CHUNK_SIZE == 50
    assert contracts.HISTORY_LEN == 4
    assert contracts.STATE45 == 45
    assert contracts.ACTION23 == 23
    assert contracts.PROPRIO256 == 256
    assert contracts.JOINT22 == 22
    assert contracts.STATE61 == 61
    assert contracts.RAW623 == 623


def test_execution23_slices_cover_23():
    slices = contracts.EXECUTION23_SLICES
    covered = list(range(23))
    for sl in slices.values():
        covered = [i for i in covered if not (sl.start <= i < sl.stop)]
    assert covered == []
    assert slices["left_pose6"] == slice(0, 6)
    assert slices["right_pose6"] == slice(6, 12)
    assert slices["continuous_gripper"] == slice(12, 14)
    assert slices["base_move3"] == slice(14, 17)
    assert slices["chest_pose6"] == slice(17, 23)


def test_gates_cannot_be_loosened():
    assert contracts.TRACKING_P95_MAX_M == 0.02
    assert contracts.TRACKING_MAX_M == 0.10
    assert contracts.GRIPPER_RANGE_M == 0.05


def test_image_contract():
    assert contracts.HEAD_RESOLUTION == (720, 720)
    assert contracts.WRIST_RESOLUTION == (480, 480)
    assert contracts.HEAD_APERTURE_DEG == 40.0


def test_2026_entry_points_at_eval_eval():
    assert contracts.ENTRY_2026["module"] == "omnigibson.eval.eval"
    assert contracts.ENTRY_2026["codec"] == "msgpack"
    assert contracts.ENTRY_2026["default_port"] == 8000


def test_wire_keys_match_post1():
    assert contracts.ACTION_CHUNK_REQUEST_KEY == "__action_chunk_size__"
    assert contracts.ACTION_KEY == "action"
    assert contracts.ACTION_CHUNK_KEY == "action_chunk"
    assert contracts.RESET_KEY == "reset"
    assert contracts.NDARRAY_KEY == b"__ndarray__"
