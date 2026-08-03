import json
from contextlib import nullcontext

import numpy as np
import pytest

from omnigibson.learning.transition_state_restore import (
    SPEC_INVALID_COMPONENT_DEPENDENCY,
    STATE_INVALID_RESTORE_COMPONENTS,
    STATE_INVALID_ROLLOUT_COMPONENTS,
    TRANSITION_STATE_CACHE_SCHEMA_VERSION,
    TransitionManifestError,
    build_restore_component_validity,
    classify_segment_component_evaluability,
    materialize_transition_events,
    metric_component_dependencies,
    normalize_transition_manifest,
    require_exact_cache_artifacts,
    restore_method_is_exact,
    restore_recorded_scene_base,
    transition_events_before_state_frame,
    validate_serialized_state_consumption,
)
from omnigibson.utils.scene_restore_utils import detach_system_template_aliases_for_restore


class FakeObject:
    def __init__(self, name, uuid, category="generated", prim_path=None):
        self.name = name
        self.uuid = uuid
        self.category = category
        self.prim_path = prim_path or f"/World/scene_0/{name}"
        self.position = None

    def set_position(self, position):
        self.position = position


FakeObject.__module__ = "fake.objects"


class WrongObject(FakeObject):
    pass


WrongObject.__module__ = "fake.objects"


class FakeSystem:
    def __init__(self, name, particle_template=None):
        self.name = name
        self._particle_template = particle_template


class FakeMaterial:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.users = set()
        self.removed = False

    def remove_user(self, user):
        self.users.remove(user)
        if not self.users:
            self.removed = True


class FakeXform:
    def __init__(self, material=None):
        self._material = material
        self.collision_meshes = {}
        self.visual_meshes = {}
        if material is not None:
            material.users.add(self)


class FakeRegistry:
    def __init__(self, objects=()):
        self.objects = list(objects)

    def __call__(self, key, value):
        for obj in self.objects:
            if getattr(obj, key) == value:
                return obj
        return None

    def remove(self, obj):
        self.objects.remove(obj)

    def object_is_registered(self, obj):
        return obj in self.objects


class FakeScene:
    def __init__(self, objects=(), systems=()):
        self.object_registry = FakeRegistry(objects)
        self.system_registry = FakeRegistry(systems)
        self.operations = []
        self._initial_file = {"sentinel": "current-evaluator-initial-file"}

    @property
    def objects(self):
        return self.object_registry.objects

    @property
    def active_systems(self):
        return {system.name: system for system in self.system_registry.objects}

    @property
    def n_objects(self):
        return len(self.objects)

    def restore(self, scene_file, update_initial_file=False):
        detached = detach_system_template_aliases_for_restore(
            scene=self,
            scene_info=scene_file,
        )
        self.operations.append(("restore_recorded_scene", scene_file, update_initial_file))
        return {"detached_system_template_aliases": detached}

    def reset(self, hard=True):
        pytest.fail("exact restore must not use the evaluator's current initial scene")

    def get_system(self, name, force_init=True):
        assert force_init
        self.operations.append(("add_system", name))
        system = FakeSystem(name)
        self.system_registry.objects.append(system)
        return system

    def clear_system(self, name):
        self.operations.append(("remove_system", name))
        system = self.system_registry("name", name)
        self.system_registry.objects.remove(system)

    def remove_object(self, obj):
        self.operations.append(("remove_object", obj.name))
        self.object_registry.objects.remove(obj)

    def add_object(self, obj):
        self.operations.append(("add_object", obj.name))
        self.object_registry.objects.append(obj)


def uuid_from_name(name):
    return {
        "whole": 101,
        "half_whole_0": 202,
        "half_whole_1": 303,
    }.get(name, 999)


def object_init(name="half_whole_0", class_name="FakeObject"):
    return {
        "class_module": "fake.objects",
        "class_name": class_name,
        "args": {"name": name, "category": "half_whole", "model": "model_a"},
    }


def event(*, add_objects=(), remove_objects=(), add_systems=(), remove_systems=()):
    return {
        "systems": {"add": list(add_systems), "remove": list(remove_systems)},
        "objects": {"add": list(add_objects), "remove": list(remove_objects)},
    }


def recorded_scene_snapshot():
    return {
        "init_info": {"class_name": "FakeScene"},
        "state": {
            "registry": {
                "object_registry": {"whole": {}},
                "system_registry": {},
            }
        },
        "objects_info": {"init_info": {"whole": {}}},
    }


def test_recorded_scene_restored_before_transitions_without_current_initial_scene():
    whole = FakeObject("whole", uuid_from_name("whole"), category="whole")
    scene = FakeScene(objects=[whole])
    snapshot = recorded_scene_snapshot()

    summary = restore_recorded_scene_base(
        scene=scene,
        raw_scene_file=json.dumps(snapshot),
        init_metadata={"recorded_marker": np.asarray([7], dtype=np.int64)},
        stopped_context=nullcontext,
    )
    materialize_transition_events(
        scene=scene,
        events=transition_events_before_state_frame(
            normalize_transition_manifest(
                {"5": event(remove_objects=["whole"], add_objects=[object_init()])}
            ),
            6,
        ),
        create_object_from_init_info=lambda info: FakeObject(
            info["args"]["name"], uuid_from_name(info["args"]["name"])
        ),
        uuid_from_name=uuid_from_name,
        park_object=lambda obj, slot: None,
        initialize_event=lambda: None,
    )

    assert scene.operations[0][0] == "restore_recorded_scene"
    assert scene.operations[0][1] == snapshot
    assert scene.operations[0][1] is not scene._initial_file
    assert scene.operations[0][2] is True
    assert whole.recorded_marker == 7
    assert summary["recorded_scene_object_count"] == 1
    assert [operation[0] for operation in scene.operations] == [
        "restore_recorded_scene",
        "remove_object",
        "add_object",
    ]


def test_recorded_scene_restore_detaches_duplicate_system_template_alias_before_restore():
    prim_path = "/World/scene_0/diced__whole/template"
    internal_template = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    registered_alias = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    system = FakeSystem("diced__whole", particle_template=internal_template)
    whole = FakeObject("whole", uuid_from_name("whole"), category="whole")

    class OrderCheckingScene(FakeScene):
        def restore(self, scene_file, update_initial_file=False):
            summary = super().restore(scene_file, update_initial_file=update_initial_file)
            assert self.object_registry("name", registered_alias.name) is None
            return summary

    scene = OrderCheckingScene(objects=[whole, registered_alias], systems=[system])
    summary = restore_recorded_scene_base(
        scene=scene,
        raw_scene_file=recorded_scene_snapshot(),
        init_metadata={},
        stopped_context=nullcontext,
    )

    assert scene.object_registry.objects == [whole]
    assert summary["detached_system_template_aliases"] == [
        {
            "system_name": "diced__whole",
            "object_name": "diced__whole_template",
            "prim_path": prim_path,
            "released_material_prim_paths": [],
        }
    ]


def test_duplicate_system_template_alias_releases_only_shared_alias_material_users():
    prim_path = "/World/scene_0/diced__whole/template"
    material_path = f"{prim_path}/Looks/material"
    material = FakeMaterial(material_path)
    internal_mesh = FakeXform(material)
    alias_mesh = FakeXform(material)
    internal_link = FakeXform()
    internal_link.visual_meshes["mesh"] = internal_mesh
    alias_link = FakeXform()
    alias_link.visual_meshes["mesh"] = alias_mesh
    internal_template = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    internal_template.links = {"base_link": internal_link}
    registered_alias = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    registered_alias.links = {"base_link": alias_link}
    system = FakeSystem("diced__whole", particle_template=internal_template)
    scene = FakeScene(objects=[registered_alias], systems=[system])

    summary = restore_recorded_scene_base(
        scene=scene,
        raw_scene_file=recorded_scene_snapshot(),
        init_metadata={},
        stopped_context=nullcontext,
    )

    assert material.users == {internal_mesh}
    assert alias_mesh._material is None
    assert not material.removed
    assert summary["detached_system_template_aliases"][0]["released_material_prim_paths"] == [
        material_path
    ]


def test_recorded_scene_restore_rejects_unshared_alias_material_user():
    prim_path = "/World/scene_0/diced__whole/template"
    material = FakeMaterial(f"{prim_path}/Looks/material")
    alias_mesh = FakeXform(material)
    internal_template = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    internal_template.links = {"base_link": FakeXform()}
    registered_alias = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    alias_link = FakeXform()
    alias_link.visual_meshes["mesh"] = alias_mesh
    registered_alias.links = {"base_link": alias_link}
    system = FakeSystem("diced__whole", particle_template=internal_template)
    scene = FakeScene(objects=[registered_alias], systems=[system])

    with pytest.raises(TransitionManifestError) as exc_info:
        restore_recorded_scene_base(
            scene=scene,
            raw_scene_file=recorded_scene_snapshot(),
            init_metadata={},
            stopped_context=nullcontext,
        )

    assert exc_info.value.reason == "system_template_alias_material_not_shared_with_owner"
    assert scene.object_registry.objects == [registered_alias]
    assert material.users == {alias_mesh}
    assert scene.operations == []


def test_alias_material_validation_is_transactional_across_multiple_xforms():
    prim_path = "/World/scene_0/diced__whole/template"
    shared_material = FakeMaterial(f"{prim_path}/Looks/shared")
    invalid_material = FakeMaterial(f"{prim_path}/Looks/unshared")
    internal_shared_mesh = FakeXform(shared_material)
    alias_shared_mesh = FakeXform(shared_material)
    alias_invalid_mesh = FakeXform(invalid_material)

    internal_link = FakeXform()
    internal_link.visual_meshes["shared"] = internal_shared_mesh
    alias_link = FakeXform()
    alias_link.visual_meshes["shared"] = alias_shared_mesh
    alias_link.visual_meshes["invalid"] = alias_invalid_mesh
    internal_template = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    internal_template.links = {"base_link": internal_link}
    registered_alias = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    registered_alias.links = {"base_link": alias_link}
    system = FakeSystem("diced__whole", particle_template=internal_template)
    scene = FakeScene(objects=[registered_alias], systems=[system])

    with pytest.raises(TransitionManifestError) as exc_info:
        restore_recorded_scene_base(
            scene=scene,
            raw_scene_file=recorded_scene_snapshot(),
            init_metadata={},
            stopped_context=nullcontext,
        )

    assert exc_info.value.reason == "system_template_alias_material_not_shared_with_owner"
    assert shared_material.users == {internal_shared_mesh, alias_shared_mesh}
    assert invalid_material.users == {alias_invalid_mesh}
    assert alias_shared_mesh._material is shared_material
    assert alias_invalid_mesh._material is invalid_material
    assert scene.object_registry.objects == [registered_alias]
    assert scene.operations == []


def test_recorded_scene_restore_rejects_template_alias_required_by_target_snapshot():
    prim_path = "/World/scene_0/diced__whole/template"
    internal_template = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    registered_alias = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    system = FakeSystem("diced__whole", particle_template=internal_template)
    snapshot = recorded_scene_snapshot()
    snapshot["state"]["registry"]["object_registry"][registered_alias.name] = {}
    snapshot["objects_info"]["init_info"][registered_alias.name] = {}
    scene = FakeScene(objects=[registered_alias], systems=[system])

    with pytest.raises(TransitionManifestError) as exc_info:
        restore_recorded_scene_base(
            scene=scene,
            raw_scene_file=snapshot,
            init_metadata={},
            stopped_context=nullcontext,
        )

    assert exc_info.value.reason == "recorded_scene_target_contains_system_template_alias"
    assert scene.object_registry.objects == [registered_alias]
    assert scene.operations == []


def test_recorded_scene_restore_rejects_template_alias_in_state_registry_only():
    prim_path = "/World/scene_0/diced__whole/template"
    internal_template = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    registered_alias = FakeObject("diced__whole_template", 404, prim_path=prim_path)
    system = FakeSystem("diced__whole", particle_template=internal_template)
    snapshot = recorded_scene_snapshot()
    snapshot["state"]["registry"]["object_registry"][registered_alias.name] = {}
    assert registered_alias.name not in snapshot["objects_info"]["init_info"]
    scene = FakeScene(objects=[registered_alias], systems=[system])

    with pytest.raises(TransitionManifestError) as exc_info:
        restore_recorded_scene_base(
            scene=scene,
            raw_scene_file=snapshot,
            init_metadata={},
            stopped_context=nullcontext,
        )

    assert exc_info.value.reason == "recorded_scene_target_contains_system_template_alias"
    assert scene.object_registry.objects == [registered_alias]
    assert scene.operations == []


def test_recorded_scene_restore_rejects_ambiguous_system_template_alias_identity():
    internal_template = FakeObject(
        "diced__whole_template",
        404,
        prim_path="/World/scene_0/diced__whole/template",
    )
    registered_alias = FakeObject(
        "diced__whole_template",
        404,
        prim_path="/World/scene_0/different/template",
    )
    system = FakeSystem("diced__whole", particle_template=internal_template)
    scene = FakeScene(objects=[registered_alias], systems=[system])

    with pytest.raises(TransitionManifestError) as exc_info:
        restore_recorded_scene_base(
            scene=scene,
            raw_scene_file=recorded_scene_snapshot(),
            init_metadata={},
            stopped_context=nullcontext,
        )

    assert exc_info.value.reason == "system_template_alias_identity_mismatch"
    assert scene.object_registry.objects == [registered_alias]
    assert scene.operations == []


def test_duplicate_system_template_alias_cleanup_is_repeatable_across_restores():
    prim_path = "/World/scene_0/diced__whole/template"
    whole = FakeObject("whole", uuid_from_name("whole"), category="whole")
    scene = FakeScene(objects=[whole])
    summaries = []

    for _ in range(2):
        internal_template = FakeObject("diced__whole_template", 404, prim_path=prim_path)
        registered_alias = FakeObject("diced__whole_template", 404, prim_path=prim_path)
        scene.system_registry.objects[:] = [
            FakeSystem("diced__whole", particle_template=internal_template)
        ]
        scene.object_registry.objects.append(registered_alias)
        summaries.append(
            restore_recorded_scene_base(
                scene=scene,
                raw_scene_file=recorded_scene_snapshot(),
                init_metadata={},
                stopped_context=nullcontext,
            )
        )
        # Fake Scene.restore records the call but does not clear systems. Model the
        # completed clean-scene synchronization before replaying the next late state.
        scene.system_registry.objects.clear()

    assert scene.object_registry.objects == [whole]
    assert [summary["detached_system_template_aliases"] for summary in summaries] == [
        [
            {
                "system_name": "diced__whole",
                "object_name": "diced__whole_template",
                "prim_path": prim_path,
                "released_material_prim_paths": [],
            }
        ],
        [
            {
                "system_name": "diced__whole",
                "object_name": "diced__whole_template",
                "prim_path": prim_path,
                "released_material_prim_paths": [],
            }
        ],
    ]


@pytest.mark.parametrize(
    ("raw_scene_file", "reason"),
    [
        (None, "recorded_scene_snapshot_missing"),
        ("{not-json", "recorded_scene_snapshot_invalid_json"),
        ({"state": {}}, "recorded_scene_snapshot_missing_field"),
    ],
)
def test_missing_or_malformed_recorded_scene_fails_closed(raw_scene_file, reason):
    scene = FakeScene(objects=[FakeObject("whole", uuid_from_name("whole"))])

    with pytest.raises(TransitionManifestError) as exc_info:
        restore_recorded_scene_base(
            scene=scene,
            raw_scene_file=raw_scene_file,
            init_metadata={},
            stopped_context=nullcontext,
        )

    assert exc_info.value.reason == reason
    assert scene.operations == []


def test_exact_cache_requires_complete_schema_v3_artifacts():
    complete_cache = {
        "schema_version": TRANSITION_STATE_CACHE_SCHEMA_VERSION,
        "transition_manifest_present": True,
        "recorded_scene_file_json": json.dumps(recorded_scene_snapshot()),
        "init_metadata": {},
    }
    require_exact_cache_artifacts(complete_cache)

    with pytest.raises(TransitionManifestError) as exc_info:
        require_exact_cache_artifacts({**complete_cache, "schema_version": 2})
    assert exc_info.value.reason == "exact_cache_schema_v3_required"

    with pytest.raises(TransitionManifestError) as exc_info:
        require_exact_cache_artifacts({**complete_cache, "recorded_scene_file_json": None})
    assert exc_info.value.reason == "recorded_scene_snapshot_missing"


def test_manifest_uses_official_state_frame_offset():
    manifest = normalize_transition_manifest(
        json.dumps(
            {
                "5": event(add_objects=[object_init()]),
                "8": event(remove_objects=["half_whole_0"]),
            }
        )
    )

    assert transition_events_before_state_frame(manifest, 5) == ()
    assert [step for step, _ in transition_events_before_state_frame(manifest, 6)] == [5]
    assert [step for step, _ in transition_events_before_state_frame(manifest, 8)] == [5]
    assert [step for step, _ in transition_events_before_state_frame(manifest, 9)] == [5, 8]


def test_dynamic_objects_are_registered_before_serialized_lookup():
    whole = FakeObject("whole", uuid_from_name("whole"), category="whole")
    scene = FakeScene(objects=[whole])
    manifest = normalize_transition_manifest(
        {
            "5": event(
                add_objects=[object_init()],
                remove_objects=["whole"],
                add_systems=["diced__whole"],
            )
        }
    )
    operations = []

    def create_object(init_info):
        operations.append(("create", init_info["args"]["name"]))
        return FakeObject(init_info["args"]["name"], uuid_from_name(init_info["args"]["name"]))

    summary = materialize_transition_events(
        scene=scene,
        events=transition_events_before_state_frame(manifest, 6),
        create_object_from_init_info=create_object,
        uuid_from_name=uuid_from_name,
        park_object=lambda obj, index: operations.append(("park", obj.name, index)),
        initialize_event=lambda: operations.append(("initialize",)),
    )

    # This models Registry.deserialize's UUID lookup and must happen only after
    # constructor replay, scene registration, parking, and initialization.
    restored_obj = scene.object_registry("uuid", uuid_from_name("half_whole_0"))
    operations.append(("serialized_lookup", restored_obj.name if restored_obj else None))

    assert restored_obj is not None
    assert scene.object_registry("name", "whole") is None
    assert scene.system_registry("name", "diced__whole") is not None
    assert operations[-2:] == [("initialize",), ("serialized_lookup", "half_whole_0")]
    assert summary["added_objects"] == ["half_whole_0"]
    assert summary["removed_objects"] == ["whole"]


def test_additions_across_transition_steps_use_distinct_global_park_slots():
    scene = FakeScene()
    manifest = normalize_transition_manifest(
        {
            "5": event(add_objects=[object_init(name="half_whole_0")]),
            "8": event(add_objects=[object_init(name="half_whole_1")]),
        }
    )
    park_calls = []

    materialize_transition_events(
        scene=scene,
        events=transition_events_before_state_frame(manifest, 9),
        create_object_from_init_info=lambda info: FakeObject(
            info["args"]["name"], uuid_from_name(info["args"]["name"])
        ),
        uuid_from_name=uuid_from_name,
        park_object=lambda obj, slot: park_calls.append((obj.name, slot)),
        initialize_event=lambda: None,
    )

    assert park_calls == [("half_whole_0", 0), ("half_whole_1", 1)]


@pytest.mark.parametrize(
    ("scene_objects", "creator", "reason"),
    [
        (
            [FakeObject("half_whole_0", uuid_from_name("half_whole_0"))],
            lambda info: FakeObject(info["args"]["name"], uuid_from_name(info["args"]["name"])),
            "transition_object_name_collision",
        ),
        (
            [FakeObject("different_name", uuid_from_name("half_whole_0"))],
            lambda info: FakeObject(info["args"]["name"], uuid_from_name(info["args"]["name"])),
            "transition_object_uuid_collision",
        ),
        (
            [],
            lambda info: WrongObject(info["args"]["name"], uuid_from_name(info["args"]["name"])),
            "transition_object_type_mismatch",
        ),
    ],
)
def test_identity_collisions_fail_closed(scene_objects, creator, reason):
    scene = FakeScene(objects=scene_objects)
    manifest = normalize_transition_manifest({"0": event(add_objects=[object_init()])})

    with pytest.raises(TransitionManifestError, match=reason) as exc_info:
        materialize_transition_events(
            scene=scene,
            events=transition_events_before_state_frame(manifest, 1),
            create_object_from_init_info=creator,
            uuid_from_name=uuid_from_name,
            park_object=lambda obj, index: None,
            initialize_event=lambda: None,
        )

    assert exc_info.value.reason == reason


def test_missing_removal_fails_closed():
    scene = FakeScene()
    manifest = normalize_transition_manifest({"0": event(remove_objects=["whole"])})

    with pytest.raises(TransitionManifestError) as exc_info:
        materialize_transition_events(
            scene=scene,
            events=transition_events_before_state_frame(manifest, 1),
            create_object_from_init_info=lambda info: None,
            uuid_from_name=uuid_from_name,
            park_object=lambda obj, index: None,
            initialize_event=lambda: None,
        )

    assert exc_info.value.reason == "transition_object_removal_missing"


def test_static_scene_manifest_is_noop():
    static_obj = FakeObject("whole", uuid_from_name("whole"), category="whole")
    scene = FakeScene(objects=[static_obj])

    summary = materialize_transition_events(
        scene=scene,
        events=transition_events_before_state_frame(normalize_transition_manifest({}), 100),
        create_object_from_init_info=lambda info: pytest.fail("static scene must not create objects"),
        uuid_from_name=uuid_from_name,
        park_object=lambda obj, index: pytest.fail("static scene must not park objects"),
        initialize_event=lambda: pytest.fail("static scene must not initialize transition events"),
    )

    assert scene.object_registry.objects == [static_obj]
    assert summary["applied_transition_steps"] == []


def test_robot_only_restore_is_never_exact():
    assert restore_method_is_exact("rawdata", {"exact_world_state": True})
    assert restore_method_is_exact("cache", {"exact_world_state": True})
    assert not restore_method_is_exact("robot", {"exact_world_state": True})
    assert not restore_method_is_exact("rawdata", {"exact_world_state": False})


def _restore_debug(validity):
    return {
        "selected_method": "rawdata",
        "component_validity": validity,
        "rawdata": {"component_validity": validity},
    }


def test_official_eval_asset_compatibility_is_separate_from_historical_exactness():
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=False,
        particle_state_valid=False,
        historical_asset_identity_verified=False,
        source_assisted_grasp_state_present=False,
        source_system_count=2,
    )

    assert validity["official_policy_eval"] == {
        "asset_compatible": True,
        "asset_policy": "installed_current_assets",
        "recorded_asset_md5_required": False,
    }
    assert validity["components"]["official_eval_asset_compatibility"]["valid"] is True
    assert validity["components"]["assisted_grasp_state"]["valid"] is False
    assert validity["components"]["particle_state"]["valid"] is False
    assert validity["historical_asset_identity_verified"] is False
    assert validity["historical_world_state_exact"] is False


def test_historical_exactness_requires_all_components_and_asset_identity():
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=True,
        particle_state_valid=True,
        historical_asset_identity_verified=True,
        source_assisted_grasp_state_present=True,
        source_system_count=1,
    )

    assert validity["historical_world_state_exact"] is True


@pytest.mark.parametrize(
    ("spec", "extra_component"),
    [
        ({"metric_type": "predicate", "name": "grasped"}, "assisted_grasp_state"),
        ({"metric_type": "predicate", "name": "covered"}, "particle_state"),
        ({"metric_type": "predicate", "name": "touching"}, "contact_state"),
    ],
)
def test_metric_dependency_mapping_is_explicit(spec, extra_component):
    row = metric_component_dependencies(spec)

    assert row["required_components"] == sorted(
        {"official_eval_asset_compatibility", "rigid_state", extra_component}
    )
    assert row["dependency_ambiguity"] is None


def test_diagnostic_grasp_dependency_does_not_block_rigid_primary_metric():
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=False,
        particle_state_valid=False,
        historical_asset_identity_verified=False,
    )
    debug = _restore_debug(validity)

    result = classify_segment_component_evaluability(
        [
            {
                "metric_type": "predicate",
                "name": "inside",
                "diagnostic_specs": [{"metric_type": "predicate", "name": "grasped"}],
            }
        ],
        {"start": debug, "end": debug, "rollout_start": debug},
    )

    assert result["boundary_evaluable"] is True
    assert "assisted_grasp_state" not in result["boundary_required_components"]
    assert result["rollout_evaluable"] is False
    assert result["rollout_missing_components"] == ["assisted_grasp_state"]


def test_unaffected_rigid_boundary_is_evaluable_but_rollout_requires_ag():
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=False,
        particle_state_valid=False,
        historical_asset_identity_verified=False,
        source_assisted_grasp_state_present=False,
        source_system_count=2,
    )
    debug = _restore_debug(validity)

    result = classify_segment_component_evaluability(
        [{"metric_type": "predicate", "name": "open"}],
        {"start": debug, "end": debug, "rollout_start": debug},
    )

    assert result["boundary_evaluable"] is True
    assert result["boundary_aggregation_eligible"] is True
    assert result["boundary_missing_components"] == []
    assert result["rollout_evaluable"] is False
    assert result["rollout_result_type"] == STATE_INVALID_ROLLOUT_COMPONENTS
    assert result["aggregation_eligible"] is False
    assert result["model_failure_eligible"] is False
    assert result["rollout_missing_components"] == ["assisted_grasp_state"]


def test_rigid_rollout_is_evaluable_when_assisted_grasp_state_is_valid():
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=True,
        particle_state_valid=False,
        source_assisted_grasp_state_present=True,
        historical_asset_identity_verified=False,
    )
    debug = _restore_debug(validity)

    result = classify_segment_component_evaluability(
        [{"metric_type": "predicate", "name": "open"}],
        {"start": debug, "end": debug, "rollout_start": debug},
    )

    assert result["boundary_evaluable"] is True
    assert result["rollout_evaluable"] is True
    assert result["aggregation_eligible"] is True
    assert result["model_failure_eligible"] is True


@pytest.mark.parametrize(
    ("predicate_name", "missing_component"),
    [
        ("grasped", "assisted_grasp_state"),
        ("covered", "particle_state"),
    ],
)
def test_missing_restore_component_is_state_invalid_not_model_failure(
    predicate_name, missing_component
):
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=False,
        particle_state_valid=False,
        historical_asset_identity_verified=False,
        source_assisted_grasp_state_present=False,
        source_system_count=2,
    )
    debug = _restore_debug(validity)

    result = classify_segment_component_evaluability(
        [{"metric_type": "predicate", "name": predicate_name}],
        {"start": debug, "end": debug, "rollout_start": debug},
    )

    assert result["evaluable"] is False
    assert result["result_type"] == STATE_INVALID_RESTORE_COMPONENTS
    assert result["aggregation_eligible"] is False
    assert result["model_failure_eligible"] is False
    assert missing_component in result["missing_components"]


def test_transfer_payload_dependency_ambiguity_fails_closed():
    row = metric_component_dependencies(
        {
            "metric_type": "predicate",
            "name": "inside",
            "params": {"metric_family": "relation_transfer_proxy"},
        }
    )

    assert row["dependency_ambiguity"] == (
        "relation_transfer_payload_rigid_or_particle_dependency_requires_runtime_binding"
    )


def test_ambiguous_effect_dependency_fails_as_spec_invalid():
    validity = build_restore_component_validity(
        restored=True,
        rigid_state_valid=True,
        assisted_grasp_state_valid=True,
        particle_state_valid=True,
        historical_asset_identity_verified=False,
    )
    debug = _restore_debug(validity)

    result = classify_segment_component_evaluability(
        [{"metric_type": "predicate", "name": "on_fire"}],
        {"start": debug, "end": debug, "rollout_start": debug},
    )

    assert result["evaluable"] is False
    assert result["result_type"] == SPEC_INVALID_COMPONENT_DEPENDENCY
    assert result["dependency_ambiguities"] == [
        "on_fire_effect_state_dependency_not_yet_classified"
    ]
    assert result["model_failure_eligible"] is False


def test_serialized_state_consumption_must_cover_complete_vector():
    telemetry = validate_serialized_state_consumption(consumed=145, total=145)

    assert telemetry == {
        "consumed": 145,
        "total": 145,
        "complete": True,
        "unconsumed": 0,
    }


def test_unconsumed_serialized_tail_fails_before_component_validity():
    with pytest.raises(TransitionManifestError) as exc_info:
        validate_serialized_state_consumption(consumed=123, total=145)

    assert exc_info.value.reason == "serialized_source_state_not_fully_consumed"
    assert exc_info.value.details == {"consumed": 123, "total": 145, "unconsumed": 22}


def test_malformed_constructor_manifest_fails_before_mutation():
    with pytest.raises(TransitionManifestError) as exc_info:
        normalize_transition_manifest(
            {
                "0": event(
                    add_objects=[
                        {
                            "class_module": "fake.objects",
                            "class_name": "FakeObject",
                            "args": {"category": "half_whole"},
                        }
                    ]
                )
            }
        )

    assert exc_info.value.reason == "transition_manifest_missing_object_name"
