"""Extract per-primitive serialized world states for BEHAVIOR-1K demos.

This utility reads primitive annotations from the *demo* dataset (2025-challenge-demos)
then extracts the *serialized* simulator state from the corresponding raw HDF5
(2025-challenge-rawdata) at each primitive boundary (by default: primitive start frame).

The output is a compact cache file per demo:
  <demo_data_path>/meta/primitive_states/task-XXXX/episode_YYYYYYYY.npz

These caches can be used by
  OmniGibson/omnigibson/learning/subtask_eval.py
when rawdata is not available at evaluation time.

Example:
  python OmniGibson/scripts/learning/extract_primitive_world_states.py \
    --demo_data_path /path/to/2025-challenge-demos \
    --rawdata_path /path/to/2025-challenge-rawdata \
    --task_name turning_on_radio \
    --demo_id 00000010 # Optional; if omitted, process multiple demos

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from omnigibson.learning.transition_state_restore import (
    TRANSITION_STATE_CACHE_SCHEMA_VERSION,
    normalize_recorded_scene_snapshot,
    normalize_transition_manifest,
)
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES


def _find_raw_hdf5(rawdata_path: str, task_idx: int, demo_id: str) -> Optional[str]:
    task_folder = f"task-{task_idx:04d}"
    candidates = [
        os.path.join(rawdata_path, task_folder, f"episode_{demo_id}.hdf5"),
        os.path.join(rawdata_path, f"episode_{demo_id}.hdf5"),
        os.path.join(rawdata_path, task_folder, f"{demo_id}.hdf5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _list_demo_ids(demo_data_path: str, task_idx: int, limit: Optional[int] = None) -> List[str]:
    task_folder = f"task-{task_idx:04d}"
    annotations_dir = os.path.join(demo_data_path, "annotations", task_folder)
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(f"Annotations folder not found: {annotations_dir}")

    demo_ids: List[str] = []
    for fname in sorted(os.listdir(annotations_dir)):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith("episode_"):
            continue
        demo_ids.append(fname.replace("episode_", "").replace(".json", ""))

    if limit is not None:
        demo_ids = demo_ids[:limit]
    return demo_ids


def _load_primitives(demo_data_path: str, task_idx: int, demo_id: str) -> Sequence[Dict]:
    task_folder = f"task-{task_idx:04d}"
    annotation_path = os.path.join(demo_data_path, "annotations", task_folder, f"episode_{demo_id}.json")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    primitives = annotations.get("primitive_annotation", [])
    if not isinstance(primitives, list):
        raise ValueError("Expected annotations['primitive_annotation'] to be a list")
    return primitives


def _primitive_frames(primitives: Sequence[Dict], include_ends: bool) -> List[int]:
    frames: List[int] = []
    for p in primitives:
        if "frame_duration" not in p:
            continue
        start_end = p["frame_duration"]
        if not isinstance(start_end, (list, tuple)) or len(start_end) != 2:
            continue
        start, end = int(start_end[0]), int(start_end[1])
        frames.append(start)
        if include_ends:
            frames.append(end)

    # unique + sorted
    frames = sorted(set(frames))
    return frames


def _pack_variable_length_vectors(vectors: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = np.array([int(v.size) for v in vectors], dtype=np.int64)
    offsets = np.zeros(len(vectors), dtype=np.int64)
    if len(vectors) > 0:
        offsets[1:] = np.cumsum(sizes[:-1])
    flat = np.concatenate([v.reshape(-1) for v in vectors], axis=0) if len(vectors) > 0 else np.zeros((0,), dtype=np.float32)
    return sizes, offsets, flat


def extract_demo(
    demo_data_path: str,
    rawdata_path: str,
    task_name: str,
    demo_id: str,
    include_ends: bool,
    output_dir: Optional[str],
) -> str:
    task_idx = TASK_NAMES_TO_INDICES[task_name]

    primitives = _load_primitives(demo_data_path, task_idx, demo_id)
    frames = _primitive_frames(primitives, include_ends=include_ends)
    if not frames:
        raise ValueError(f"No primitive frames found for demo {demo_id} (task={task_name})")

    raw_hdf5_path = _find_raw_hdf5(rawdata_path, task_idx, demo_id)
    if raw_hdf5_path is None:
        raise FileNotFoundError(f"Raw HDF5 not found for demo {demo_id} under {rawdata_path}")

    task_folder = f"task-{task_idx:04d}"
    if output_dir is None:
        out_root = os.path.join(demo_data_path, "meta", "primitive_states")
    else:
        out_root = output_dir
    out_path = os.path.join(out_root, task_folder, f"episode_{demo_id}.npz")
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    # Read raw state vectors
    with h5py.File(raw_hdf5_path, "r") as f:
        data_grp = f["data"]
        demo_keys = [k for k in data_grp.keys() if k.startswith("demo_")]
        if not demo_keys:
            raise ValueError(f"No demo groups found in raw file: {raw_hdf5_path}")
        max_requested_frame = max(frames)
        demo_candidates = []
        for key in demo_keys:
            grp = data_grp[key]
            if "state" not in grp or "state_size" not in grp:
                continue
            state_len = len(grp["state"])
            demo_candidates.append((max_requested_frame >= state_len, -state_len, key, grp))
        if not demo_candidates:
            raise ValueError(f"Raw file missing 'state'/'state_size': {raw_hdf5_path}")
        _, _, selected_demo_key, demo_grp = min(demo_candidates)

        state = demo_grp["state"]
        state_size = demo_grp["state_size"]
        if "transitions" not in demo_grp.attrs:
            raise ValueError(f"Raw file missing transition manifest: {raw_hdf5_path}:{selected_demo_key}")
        transition_manifest = normalize_transition_manifest(demo_grp.attrs["transitions"])
        transition_manifest_json = json.dumps(transition_manifest, sort_keys=True, separators=(",", ":"))

        if "scene_file" not in data_grp.attrs:
            raise ValueError(f"Raw file missing recorded scene_file: {raw_hdf5_path}")
        recorded_scene_file = normalize_recorded_scene_snapshot(data_grp.attrs["scene_file"])
        recorded_scene_file_json = json.dumps(recorded_scene_file, separators=(",", ":"))

        if "init_metadata" not in demo_grp:
            raise ValueError(f"Raw file missing init_metadata: {raw_hdf5_path}:{selected_demo_key}")
        init_metadata_group = demo_grp["init_metadata"]
        init_metadata_arrays: Dict[str, np.ndarray] = {}
        for key, value in init_metadata_group.items():
            if not isinstance(value, h5py.Dataset):
                raise ValueError(f"Nested init_metadata is unsupported for exact cache: {key}")
            init_metadata_arrays[key] = np.asarray(value[()])

        vectors: List[np.ndarray] = []
        kept_frames: List[int] = []
        for frame_idx in frames:
            if frame_idx < 0 or frame_idx >= len(state):
                # Skip out-of-range; annotations occasionally include end frame == T
                continue
            sz = int(state_size[frame_idx])
            vec = np.asarray(state[frame_idx, :sz])
            # Make sure we store a dense 1D float array
            vec = vec.reshape(-1)
            vectors.append(vec)
            kept_frames.append(int(frame_idx))

    sizes, offsets, flat = _pack_variable_length_vectors(vectors)

    cache_payload = {
        "schema_version": np.array(TRANSITION_STATE_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "task_name": np.array(task_name),
        "task_idx": np.array(task_idx, dtype=np.int64),
        "demo_id": np.array(demo_id),
        "frame_indices": np.array(kept_frames, dtype=np.int64),
        "state_sizes": sizes,
        "state_offsets": offsets,
        "state_flat": flat,
        "source": np.array(f"rawdata:{raw_hdf5_path}:{selected_demo_key}"),
        "include_ends": np.array(bool(include_ends)),
        "transition_manifest_json": np.array(transition_manifest_json),
        "recorded_scene_file_json": np.array(recorded_scene_file_json),
        "init_metadata_keys_json": np.array(json.dumps(list(init_metadata_arrays.keys()))),
    }
    for index, values in enumerate(init_metadata_arrays.values()):
        cache_payload[f"init_metadata_{index:04d}"] = values
    np.savez_compressed(out_path, **cache_payload)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_data_path", type=str, default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos", 
                        help="Path to 2025-challenge-demos")
    parser.add_argument("--rawdata_path", type=str, default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata", 
                        help="Path to 2025-challenge-rawdata")
    parser.add_argument("--task_name", type=str, default="turning_on_radio", 
                        help="Task name (e.g., turning_on_radio)")
    parser.add_argument("--demo_id", type=str, default=None, help="Demo id (e.g., 00000010). If omitted, process multiple.")
    parser.add_argument("--limit", type=int, default=None, help="If demo_id is omitted, limit number of demos.")
    parser.add_argument(
        "--include_ends",
        action="store_true",
        help="Also cache states at primitive end frames (in addition to start frames).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output root (default: <demo_data_path>/meta/primitive_states)",
    )

    args = parser.parse_args()
    args.include_ends = True # TODO: remove this later

    demo_data_path = os.path.expanduser(args.demo_data_path)
    rawdata_path = os.path.expanduser(args.rawdata_path)
    task_idx = TASK_NAMES_TO_INDICES[args.task_name]

    if args.demo_id is not None:
        demo_ids = [str(args.demo_id).zfill(8)]
    else:
        demo_ids = _list_demo_ids(demo_data_path, task_idx, limit=args.limit)

    print(f"Task: {args.task_name} ({task_idx}), demos: {len(demo_ids)}")

    for demo_id in demo_ids:
        out_path = extract_demo(
            demo_data_path=demo_data_path,
            rawdata_path=rawdata_path,
            task_name=args.task_name,
            demo_id=demo_id,
            include_ends=bool(args.include_ends),
            output_dir=args.output_dir,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
