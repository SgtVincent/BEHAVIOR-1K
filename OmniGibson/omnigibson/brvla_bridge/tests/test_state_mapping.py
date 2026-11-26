"""State mapping: fail-closed until a verified 61/623 -> 45 provider exists."""
from brvla_bridge import contracts, state_mapping
from brvla_bridge.errors import MappingNotDeclaredError
from brvla_bridge.tests._assert import raises


def test_unmapped_provider_rejects_mapping():
    provider = state_mapping.UnmappedProvider("demos_parquet", contracts.STATE61)
    desc = provider.descriptor()
    assert desc.source_width == 61 and desc.target_width == 45
    assert len(desc.unknown_fields()) == 45
    with raises(MappingNotDeclaredError):
        provider.map_state([0.0] * 61)


def test_raw623_unmapped_provider():
    provider = state_mapping.UnmappedProvider("rawdata_hdf5", contracts.RAW623)
    with raises(MappingNotDeclaredError):
        provider.map_state([0.0] * 623)


def test_registry_register_and_get():
    state_mapping.clear_registry()
    provider = state_mapping.UnmappedProvider("tmp", 61)
    state_mapping.register_provider("tmp", provider)
    assert state_mapping.get_provider("tmp") is provider
    state_mapping.clear_registry()


def test_registry_rejects_duplicate():
    state_mapping.clear_registry()
    state_mapping.register_provider("dup", state_mapping.UnmappedProvider("dup", 61))
    with raises(Exception):
        state_mapping.register_provider("dup", state_mapping.UnmappedProvider("dup", 61))
    state_mapping.clear_registry()


def test_registry_rejects_unknown_name():
    state_mapping.clear_registry()
    with raises(Exception):
        state_mapping.get_provider("missing")


def test_validate_mapped_state_width_and_finite():
    with raises(MappingNotDeclaredError):
        state_mapping.validate_mapped_state([0.0] * 44)
    with raises(MappingNotDeclaredError):
        state_mapping.validate_mapped_state([float("inf")] * 45)
    out = state_mapping.validate_mapped_state([0.5] * 45)
    assert len(out) == 45
