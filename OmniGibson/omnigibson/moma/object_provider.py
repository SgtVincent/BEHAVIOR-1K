"""Actual upstream MoMaGen object-centric pose retargeting, with native23 boundary.

This is a trajectory provider, NOT upstream navigation or a success evaluator.

Ported from the a28 frozen baseline ``momagen_object_provider.py`` (numpy-only).
The source-to-live SE3 transform is injected rather than imported: the MoMaGen
clean reference snapshot is ``ChengshuLi/MoMaGen@5da621667669b15b97d7a220d086283b3c8ad0e5``
(license still pending integration screening; gap in DEPENDENCY_MANIFEST). The
dirty private 44-file variant on the old sim host must not be treated as the
official artifact.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

REFERENCE_MOMAGEN_COMMIT = "5da621667669b15b97d7a220d086283b3c8ad0e5"


def rigid(pose) -> np.ndarray:
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (4, 4) or not np.isfinite(p).all():
        raise ValueError("invalid_pose_shape_or_finite")
    if not np.allclose(p[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("invalid_homogeneous_row")
    r = p[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or abs(np.linalg.det(r) - 1) > 1e-6:
        raise ValueError("invalid_rotation")
    return p.copy()


class ObjectCentricProvider:
    """Object-centric provider.

    Parameters
    ----------
    source_object_world : single 4x4 or sequence of 4x4 source object poses.
    source_left_world, source_right_world : sequence of source EEF poses.
    grippers : ``(N, 2)`` gripper targets.
    transform_fn : callable ``(live_object_pose, left_seq, source_object_pose) ->
        transformed_seq`` implementing the official object-pose segment
        transform. Injected so this package does not import MoMaGen at top level.
    """

    def __init__(self, source_object_world, source_left_world, source_right_world,
                 grippers, transform_fn: Callable[..., Any]):
        objects = np.asarray(source_object_world, dtype=np.float64)
        self.obj = rigid(objects) if objects.ndim == 2 else np.stack([rigid(p) for p in objects])
        self.left = np.stack([rigid(p) for p in source_left_world])
        self.right = np.stack([rigid(p) for p in source_right_world])
        self.grippers = np.asarray(grippers, dtype=np.float64).copy()
        if self.obj.ndim == 3 and len(self.obj) != len(self.left):
            raise ValueError("source_object_sequence_length_invalid")
        if len(self.left) != len(self.right) or self.grippers.shape != (len(self.left), 2):
            raise ValueError("source_lengths_or_grippers_invalid")
        if not np.isfinite(self.grippers).all():
            raise ValueError("gripper_nonfinite")
        if not callable(transform_fn):
            raise TypeError("transform_fn_callable_required")
        self.transform = transform_fn
        source_file: Optional[str] = None
        try:
            source_file = inspect.getfile(self.transform)
        except (TypeError, OSError):
            source_file = None
        self.source_sha256: Optional[str] = None
        if source_file is not None and source_file not in ("<stdin>", "<string>"):
            source_path = Path(source_file)
            if source_path.is_file():
                self.source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    def target(self, index: int, live_object_world) -> Dict[str, Any]:
        live = rigid(live_object_world)
        if not 0 <= index < len(self.left):
            raise IndexError("waypoint_index_out_of_range")
        source_object = self.obj if self.obj.ndim == 2 else self.obj[index]
        left = self.transform(live, self.left[index:index + 1], source_object)[0]
        right = self.transform(live, self.right[index:index + 1], source_object)[0]
        return {
            "target_pose8x4_world": np.vstack([rigid(left), rigid(right)]),
            "grippers": self.grippers[index].copy(),
            "source_index": index,
            "provider": "momagen.utils.pose_utils.transform_source_data_segment_using_object_pose",
            "provider_sha256": self.source_sha256,
            "momagen_reference_commit": REFERENCE_MOMAGEN_COMMIT,
            "live_object_pose_world": live,
            "physical_validation": False,
        }

    def action(self, index: int, live_object_world, executor: Any) -> Dict[str, Any]:
        target = self.target(index, live_object_world)
        planned = executor.plan(target)
        if not isinstance(planned, dict):
            raise ValueError("strict_executor_payload_invalid")
        action = planned.get("action23")
        vector = np.asarray(getattr(action, "vector", action), dtype=np.float64).reshape(-1)
        if vector.shape != (23,) or not np.isfinite(vector).all():
            raise ValueError("native23_invalid")
        cert = planned.get("certificate")
        if not isinstance(cert, dict) or cert.get("status") != "optimal":
            raise ValueError("strict_executor_certificate_not_optimal")
        stages = cert.get("stage_certificates")
        if stages is not None and (not isinstance(stages, (list, tuple)) or not stages or
                                    any(not isinstance(s, dict) or s.get("status") != "optimal"
                                        for s in stages)):
            raise ValueError("strict_stage_not_optimal")
        return {"action23": action, "certificate": cert, "target": target}
