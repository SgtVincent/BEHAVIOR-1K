"""CPU tests for the episode step writer and atomic finalizer."""
import json
import math

import pytest

from omnigibson.moma.episode_writer import EpisodeWriter


def test_timeline_appends_one_clean_row_per_step(tmp_path):
    writer = EpisodeWriter(tmp_path / "run-1")
    writer.record_step({"step": 0, "events": [{"type": "grasp"}]})
    writer.record_step({"step": 1, "events": []}, flush=True)
    lines = (tmp_path / "run-1" / "timeline.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 0


def test_progress_and_last_attempt_are_atomic_json(tmp_path):
    writer = EpisodeWriter(tmp_path / "run-2")
    writer.write_progress({"issued_actions": 3})
    writer.write_last_attempt({"step": 2})
    assert json.loads((tmp_path / "run-2" / "progress.json").read_text())["issued_actions"] == 3
    assert json.loads((tmp_path / "run-2" / "last_attempt.json").read_text())["step"] == 2


def test_nonfinite_value_cleaned_before_serialize(tmp_path):
    writer = EpisodeWriter(tmp_path / "run-3")
    writer.record_step({"value": math.inf})
    writer.write_progress({"value": math.nan})
    row = json.loads((tmp_path / "run-3" / "timeline.jsonl").read_text().strip())
    assert row["value"]["status"] == "UNKNOWN"
    prog = json.loads((tmp_path / "run-3" / "progress.json").read_text())
    assert prog["value"]["status"] == "UNKNOWN"


def test_finalize_writes_result_and_seals_video_ledger(tmp_path):
    writer = EpisodeWriter(tmp_path / "run-4")
    writer.record_step({"step": 0})
    report = writer.finalize({"run_id": "run-4", "closed_loop_success": False},
                            videos_sealed=["full_head.mp4"])
    result = json.loads((tmp_path / "run-4" / "result.json").read_text())
    assert result["finished_unix"] > 0
    assert result["videos_sealed"] == ["full_head.mp4"]
    assert report["videos_sealed"] == ["full_head.mp4"]


def test_double_finalize_rejected(tmp_path):
    writer = EpisodeWriter(tmp_path / "run-5")
    writer.finalize({"run_id": "run-5"})
    with pytest.raises(RuntimeError, match="already finalized"):
        writer.finalize({"run_id": "run-5"})


def test_write_after_finalize_rejected(tmp_path):
    writer = EpisodeWriter(tmp_path / "run-6")
    writer.finalize({"run_id": "run-6"})
    with pytest.raises(RuntimeError, match="finalized"):
        writer.record_step({"step": 1})
