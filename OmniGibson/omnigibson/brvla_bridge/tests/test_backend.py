"""Backend receipts: replay gate requires a resolved official cuRobo receipt."""
from brvla_bridge import backend
from brvla_bridge.errors import BackendNotDeclaredError, Port2026Error
from brvla_bridge.tests._assert import raises


def _receipt(kind, license_status, version="build-78612f45"):
    return backend.BackendReceipt(
        kind=kind,
        version=version,
        license_status=license_status,
        source_sha256="deadbeef",
        declared_by="s_0007Py1VB6a",
        notes="test receipt",
    )


def test_curobo_private_manifest_receipt_constructs():
    r = _receipt(backend.CUROBO_BACKEND, "private_manifest")
    assert r.kind == backend.CUROBO_BACKEND
    assert backend.CUROBO_PIN_REF.startswith("StanfordVL/curobo@")


def test_curobo_unknown_license_rejected():
    with raises(BackendNotDeclaredError):
        _receipt(backend.CUROBO_BACKEND, "unknown_pending")


def test_invalid_license_status_rejected():
    with raises(Port2026Error):
        _receipt(backend.STOCK_IK, "gpl-3.0")


def test_replay_gate_fails_without_curobo():
    backend.clear_registry()
    backend.register_backend(_receipt(backend.DLS_ACTIVE_SET_V2, "private_manifest"), lambda: None)
    with raises(BackendNotDeclaredError):
        backend.require_replay_backend()
    backend.clear_registry()


def test_replay_gate_rejects_curobo_without_version():
    backend.clear_registry()
    receipt = backend.BackendReceipt(
        kind=backend.CUROBO_BACKEND, version=None, license_status="private_manifest",
        source_sha256="dead", declared_by="s_0007Py1VB6a",
    )
    backend.register_backend(receipt, lambda: None)
    with raises(BackendNotDeclaredError):
        backend.require_replay_backend()
    backend.clear_registry()


def test_replay_gate_passes_with_curobo_receipt():
    backend.clear_registry()
    backend.register_backend(_receipt(backend.CUROBO_BACKEND, "private_manifest"), lambda: None)
    receipt = backend.require_replay_backend()
    assert receipt.kind == backend.CUROBO_BACKEND
    backend.clear_registry()


def test_register_rejects_non_receipt():
    backend.clear_registry()
    with raises(Port2026Error):
        backend.register_backend("not-a-receipt", lambda: None)
    backend.clear_registry()


def test_build_backend_unknown():
    backend.clear_registry()
    with raises(BackendNotDeclaredError):
        backend.build_backend("missing")
