"""Checkpoint/stats pairing lineage: recorded shapes/hashes, provenance pending."""
from types import SimpleNamespace

from brvla_bridge import lineage
from brvla_bridge.errors import Port2026Error
from brvla_bridge.tests._assert import raises


def _manifest():
    return SimpleNamespace(
        checkpoint={"index_sha256": "abc", "weight_payload_rehash": False},
        stats={
            "stats_full": {
                "sha256": "full",
                "shapes": {
                    "observation.state": {"mean": [45], "std": [45], "q01": [45], "q99": [45]},
                    "action": {"mean": [45], "std": [45], "q01": [45], "q99": [45]},
                },
                "metadata": {"schema": "b1k_full_stats_v1", "episodes": 475,
                             "producer_sha256": "prod"},
                "provenance_flag": "475 vs 800 reconciliation pending",
            },
            "stats_251101": {
                "sha256": "old",
                "shapes": {"action": {"q01": [50, 45], "q99": [50, 45]}},
            },
        },
        training_source={"dataset_members": "800 members: task0000/0001/0034/0035 x 200"},
        server_session={}, replay={}, raw={},
    )


def test_pairing_valid_and_provenance_pending():
    report = lineage.verify_stats_pairing(_manifest())
    assert report.ok
    assert report.provenance_status == "provenance_pending"
    assert report.checkpoint_index_sha == "abc"
    assert report.stats_full_sha == "full"
    assert report.stats_251101_sha == "old"
    assert any("475" in c for c in report.checks)


def test_wrong_action_shape_flagged():
    manifest = _manifest()
    manifest.stats["stats_full"]["shapes"]["action"]["mean"] = [44]
    report = lineage.verify_stats_pairing(manifest)
    assert not report.ok
    assert any("action.mean" in p for p in report.problems)
    with raises(Port2026Error):
        lineage.require_paired_stats(manifest)


def test_schema_mismatch_flagged():
    manifest = _manifest()
    manifest.stats["stats_full"]["metadata"]["schema"] = "other"
    report = lineage.verify_stats_pairing(manifest)
    assert not report.ok
    assert any("schema" in p for p in report.problems)
