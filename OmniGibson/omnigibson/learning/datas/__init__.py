from .iterable_dataset import BehaviorIterableDataset

# Optional dependency: `datasets` / `lerobot` (HuggingFace)
# Keep core dataset loaders usable even if these aren't installed.
try:
    from .lerobot_dataset import BehaviorLeRobotDataset, BehaviorLerobotDatasetMetadata
except Exception:  # noqa: BLE001
    BehaviorLeRobotDataset = None  # type: ignore[assignment]
    BehaviorLerobotDatasetMetadata = None  # type: ignore[assignment]

__all__ = [
    "BehaviorIterableDataset",
    "BehaviorLeRobotDataset",
    "BehaviorLerobotDatasetMetadata",
]