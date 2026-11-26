"""Real step writer and episode finalizer for the MoMa runner.

Ported from the a28 finalizer contract:

- every executed step is appended to ``timeline.jsonl`` and flushed at the
  same boundaries the baseline used (first step, every 30 steps, events,
  required grasps);
- ``progress.json`` carries the latest live progress;
- ``result.json`` is written atomically at finalize with the closed-loop
  report, sealed video ledger and provenance digests;
- non-finite values are cleaned before serialization; a failed step row is
  persisted separately rather than dropping the evidence.

Sim-free: stdlib only. Video containers themselves are provided by the
sim-adapter seam in the GPU stage; this writer only records the ledger.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .observer.three_can import clean


class EpisodeWriter:
    """Append-only run-directory writer with atomic final report."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._timeline = (self.out_dir / "timeline.jsonl").open("a", encoding="utf-8", buffering=1024 * 1024)
        self._closed = False
        self.videos_sealed: List[str] = []

    # -- step ledger -------------------------------------------------------

    def record_step(self, row: Dict[str, Any], *, flush: bool = False) -> None:
        if self._closed:
            raise RuntimeError("writer_finalized_no_more_writes")
        payload = clean(row)
        self._timeline.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self._timeline.flush()
        if flush:
            self._timeline.flush()

    def write_progress(self, progress: Dict[str, Any]) -> None:
        self._atomic("progress.json", clean(progress))

    def write_last_attempt(self, row: Dict[str, Any]) -> None:
        self._atomic("last_attempt.json", clean(row))

    def write_failed_step(self, row: Dict[str, Any]) -> None:
        self._atomic("failed_step.json", clean(row))

    # -- finalize ----------------------------------------------------------

    def finalize(self, report: Dict[str, Any], *, videos_sealed: Optional[List[str]] = None) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("writer already finalized; no more writes")
        if videos_sealed:
            self.videos_sealed = list(videos_sealed)
        report.setdefault("videos_sealed", list(self.videos_sealed))
        report["finished_unix"] = time.time()
        self._timeline.close()
        self._closed = True
        self._atomic("result.json", clean(report))
        return report

    # -- internals ---------------------------------------------------------

    def _atomic(self, name: str, value: Dict[str, Any]) -> None:
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        fd = os.fdopen(fd, "w", encoding="utf-8")
        try:
            json.dump(value, fd, ensure_ascii=False, indent=2, allow_nan=False)
            fd.write("\n")
            fd.flush()
            os.fsync(fd.fileno())
        finally:
            fd.close()
        os.replace(tmp, path)
