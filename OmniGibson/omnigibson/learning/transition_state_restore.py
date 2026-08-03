"""Fail-closed helpers for replaying recorded simulator lifecycle transitions.

Serialized OmniGibson states identify objects by UUID, but do not encode constructor
metadata. DataCollectionWrapper records the missing constructor metadata separately in
the HDF5 demo group's ``transitions`` attribute. These helpers validate and replay that
manifest up to a target serialized state frame before deserialization.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from omnigibson.utils.scene_restore_utils import SceneRestorePreflightError


TRANSITION_STATE_CACHE_SCHEMA_VERSION = 3
RESTORE_COMPONENT_VALIDITY_VERSION = 1
STATE_INVALID_RESTORE_COMPONENTS = "state_invalid_restore_components"
STATE_INVALID_ROLLOUT_COMPONENTS = "state_invalid_rollout_components"
SPEC_INVALID_COMPONENT_DEPENDENCY = "spec_invalid_component_dependency"

_RIGID_PREDICATES = {
    "attached",
    "inside",
    "nextto",
    "ontop",
    "open",
    "toggled_on",
    "under",
}
_PARTICLE_PREDICATES = {"contains", "covered", "filled", "saturated"}
_RIGID_METRIC_TYPES = {
    "base_to_object",
    "face_object",
    "object_orientation_match",
    "object_pose_match",
}


def restore_method_is_exact(method: str, restore_result: Optional[Mapping[str, Any]] = None) -> bool:
    """Return whether a restore result has strict historical full-world semantics.

    This is deliberately distinct from current official policy-eval compatibility.
    Robot-only proprioception is excluded even if a caller accidentally places an
    ``exact_world_state`` flag in its result payload.
    """

    return method in {"rawdata", "cache"} and bool((restore_result or {}).get("exact_world_state", False))


def build_restore_component_validity(
    *,
    restored: bool,
    rigid_state_valid: bool,
    assisted_grasp_state_valid: bool,
    particle_state_valid: bool,
    contact_state_valid: bool = False,
    historical_asset_identity_verified: bool = False,
    source_assisted_grasp_state_present: Optional[bool] = None,
    source_system_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Build component-level validity for historical mid-state restoration.

    Official policy evaluation always instantiates the installed current assets and
    does not require recorded MD5 identity. Historical asset identity remains separate
    provenance and contributes only to strict historical exactness.
    """

    official_asset_compatible = bool(restored)
    components = {
        "official_eval_asset_compatibility": {
            "valid": official_asset_compatible,
            "reason": (
                "current_installed_assets_match_official_policy_eval_semantics"
                if official_asset_compatible
                else "restore_not_completed"
            ),
            "recorded_md5_required": False,
        },
        "rigid_state": {
            "valid": bool(restored and rigid_state_valid),
            "reason": "serialized_rigid_state_loaded" if restored and rigid_state_valid else "rigid_state_unavailable",
        },
        "assisted_grasp_state": {
            "valid": bool(restored and assisted_grasp_state_valid),
            "reason": (
                "serialized_assisted_grasp_state_present"
                if restored and assisted_grasp_state_valid and source_assisted_grasp_state_present
                else "assisted_grasp_state_not_required"
                if restored and assisted_grasp_state_valid
                else "historical_assisted_grasp_state_unavailable"
            ),
        },
        "particle_state": {
            "valid": bool(restored and particle_state_valid),
            "reason": (
                "no_serialized_particle_system_state_required"
                if restored and particle_state_valid and not source_system_count
                else "serialized_particle_state_validated"
                if restored and particle_state_valid
                else "historical_particle_state_unverified"
            ),
        },
        "contact_state": {
            "valid": bool(restored and contact_state_valid),
            "reason": (
                "contact_state_validated"
                if restored and contact_state_valid
                else "post_restore_contact_state_not_validated"
            ),
        },
    }
    historical_exact = bool(
        restored
        and historical_asset_identity_verified
        and components["rigid_state"]["valid"]
        and components["assisted_grasp_state"]["valid"]
        and components["particle_state"]["valid"]
    )
    return {
        "schema_version": RESTORE_COMPONENT_VALIDITY_VERSION,
        "restore_semantics": "historical_mid_state_component_validity",
        "official_policy_eval": {
            "asset_compatible": official_asset_compatible,
            "asset_policy": "installed_current_assets",
            "recorded_asset_md5_required": False,
        },
        "components": components,
        "source_assisted_grasp_state_present": source_assisted_grasp_state_present,
        "source_system_count": source_system_count,
        "historical_asset_identity_verified": bool(historical_asset_identity_verified),
        "historical_world_state_exact": historical_exact,
    }


def _spec_field(spec: Any, name: str, default: Any = None) -> Any:
    return spec.get(name, default) if isinstance(spec, Mapping) else getattr(spec, name, default)


def metric_component_dependencies(spec: Any) -> Dict[str, Any]:
    """Map one primary segment metric to required restored-state components."""

    metric_type = str(_spec_field(spec, "metric_type", ""))
    name = str(_spec_field(spec, "name", ""))
    params = _spec_field(spec, "params", {})
    metric_family = params.get("metric_family") if isinstance(params, Mapping) else None
    dependencies = {"official_eval_asset_compatibility", "rigid_state"}
    ambiguity = None
    if metric_family == "relation_transfer_proxy":
        ambiguity = "relation_transfer_payload_rigid_or_particle_dependency_requires_runtime_binding"
    elif metric_type in _RIGID_METRIC_TYPES:
        pass
    elif metric_type == "predicate":
        if name == "grasped":
            dependencies.add("assisted_grasp_state")
        elif name in _PARTICLE_PREDICATES:
            dependencies.add("particle_state")
        elif name == "touching":
            dependencies.add("contact_state")
        elif name == "on_fire":
            ambiguity = "on_fire_effect_state_dependency_not_yet_classified"
        elif name not in _RIGID_PREDICATES:
            ambiguity = f"unmapped_predicate_dependency:{name}"
    else:
        ambiguity = f"unmapped_metric_type_dependency:{metric_type}"
    return {
        "metric_type": metric_type,
        "name": name,
        "metric_family": metric_family,
        "required_components": sorted(dependencies),
        "dependency_ambiguity": ambiguity,
    }


def _restore_component_validity_from_debug(debug: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not isinstance(debug, Mapping):
        return None
    validity = debug.get("component_validity")
    if isinstance(validity, Mapping):
        return validity
    method = debug.get("selected_method")
    selected = debug.get(method) if isinstance(method, str) else None
    if isinstance(selected, Mapping) and isinstance(selected.get("component_validity"), Mapping):
        return selected["component_validity"]
    return None


def classify_segment_component_evaluability(
    predicate_specs: Sequence[Any],
    restore_debug_by_stage: Mapping[str, Optional[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Classify boundary reads separately from rollout/model validity.

    Primary predicates require only their explicit read dependencies. Starting a policy
    rollout also requires valid assisted-grasp dynamics, even for a rigid-only predicate.
    """

    dependency_rows = [metric_component_dependencies(spec) for spec in predicate_specs]
    boundary_required = sorted(
        {component for row in dependency_rows for component in row["required_components"]}
    )
    rollout_required = sorted({*boundary_required, "assisted_grasp_state"})
    ambiguities = [row["dependency_ambiguity"] for row in dependency_rows if row["dependency_ambiguity"]]
    stage_rows = []
    boundary_missing = set()
    rollout_missing = set()
    for stage, debug in restore_debug_by_stage.items():
        validity = _restore_component_validity_from_debug(debug)
        if validity is None:
            stage_boundary_missing = list(boundary_required)
            stage_rollout_missing = list(rollout_required)
            stage_rows.append(
                {
                    "stage": stage,
                    "validity_present": False,
                    "boundary_missing_components": stage_boundary_missing,
                    "rollout_missing_components": stage_rollout_missing,
                }
            )
        else:
            components = validity.get("components", {})
            stage_boundary_missing = [
                component
                for component in boundary_required
                if not isinstance(components.get(component), Mapping)
                or not bool(components[component].get("valid"))
            ]
            stage_rollout_missing = [
                component
                for component in rollout_required
                if not isinstance(components.get(component), Mapping)
                or not bool(components[component].get("valid"))
            ]
            stage_rows.append(
                {
                    "stage": stage,
                    "validity_present": True,
                    "boundary_missing_components": stage_boundary_missing,
                    "rollout_missing_components": stage_rollout_missing,
                    "historical_world_state_exact": bool(
                        validity.get("historical_world_state_exact", False)
                    ),
                }
            )
        boundary_missing.update(stage_boundary_missing)
        rollout_missing.update(stage_rollout_missing)

    boundary_evaluable = not ambiguities and not boundary_missing
    boundary_result_type = (
        "boundary_evaluable"
        if boundary_evaluable
        else SPEC_INVALID_COMPONENT_DEPENDENCY
        if ambiguities
        else STATE_INVALID_RESTORE_COMPONENTS
    )
    rollout_evaluable = boundary_evaluable and not rollout_missing
    rollout_result_type = (
        "evaluable"
        if rollout_evaluable
        else boundary_result_type
        if not boundary_evaluable
        else STATE_INVALID_ROLLOUT_COMPONENTS
    )
    return {
        # Backward-compatible aliases refer to rollout/model evaluability.
        "evaluable": rollout_evaluable,
        "result_type": rollout_result_type,
        "aggregation_eligible": rollout_evaluable,
        "model_failure_eligible": rollout_evaluable,
        "boundary_evaluable": boundary_evaluable,
        "boundary_aggregation_eligible": boundary_evaluable,
        "boundary_result_type": boundary_result_type,
        "boundary_required_components": boundary_required,
        "boundary_missing_components": sorted(boundary_missing),
        "rollout_evaluable": rollout_evaluable,
        "rollout_result_type": rollout_result_type,
        "rollout_required_components": rollout_required,
        "rollout_missing_components": sorted(rollout_missing),
        "required_components": rollout_required,
        "missing_components": sorted(rollout_missing),
        "dependency_rows": dependency_rows,
        "dependency_ambiguities": ambiguities,
        "restore_stages": stage_rows,
    }


class TransitionManifestError(RuntimeError):
    """A recorded component-restore artifact cannot be applied safely."""

    def __init__(self, reason: str, **details: Any):
        self.reason = reason
        self.details = details
        detail_text = ", ".join(f"{key}={value!r}" for key, value in sorted(details.items()))
        super().__init__(reason if not detail_text else f"{reason}: {detail_text}")


def validate_serialized_state_consumption(*, consumed: int, total: int) -> Dict[str, Any]:
    """Fail closed unless the current schema consumes the complete serialized vector."""

    consumed = int(consumed)
    total = int(total)
    if consumed != total:
        raise TransitionManifestError(
            "serialized_source_state_not_fully_consumed",
            consumed=consumed,
            total=total,
            unconsumed=total - consumed,
        )
    return {
        "consumed": consumed,
        "total": total,
        "complete": True,
        "unconsumed": 0,
    }


def normalize_recorded_scene_snapshot(raw_scene_file: Any) -> Dict[str, Any]:
    """Decode and validate the HDF5 ``data.attrs['scene_file']`` snapshot."""

    if raw_scene_file is None:
        raise TransitionManifestError("recorded_scene_snapshot_missing")
    if hasattr(raw_scene_file, "item") and not isinstance(raw_scene_file, (str, bytes, Mapping)):
        raw_scene_file = raw_scene_file.item()
    if isinstance(raw_scene_file, bytes):
        raw_scene_file = raw_scene_file.decode("utf-8")
    if isinstance(raw_scene_file, str):
        if not raw_scene_file.strip():
            raise TransitionManifestError("recorded_scene_snapshot_missing")
        try:
            scene_file = json.loads(raw_scene_file)
        except json.JSONDecodeError as exc:
            raise TransitionManifestError(
                "recorded_scene_snapshot_invalid_json",
                line=exc.lineno,
                column=exc.colno,
            ) from exc
    elif isinstance(raw_scene_file, Mapping):
        # Scene.restore converts lists to tensors in place. A fresh deep copy keeps the
        # cached artifact immutable across non-monotonic repeated restores.
        scene_file = copy.deepcopy(dict(raw_scene_file))
    else:
        raise TransitionManifestError(
            "recorded_scene_snapshot_invalid_type",
            actual=type(raw_scene_file).__name__,
        )

    required_paths = (
        ("init_info",),
        ("state",),
        ("state", "registry"),
        ("state", "registry", "object_registry"),
        ("state", "registry", "system_registry"),
        ("objects_info",),
        ("objects_info", "init_info"),
    )
    for path in required_paths:
        value: Any = scene_file
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise TransitionManifestError(
                    "recorded_scene_snapshot_missing_field",
                    path=".".join(path),
                )
            value = value[key]
        if not isinstance(value, Mapping):
            raise TransitionManifestError(
                "recorded_scene_snapshot_invalid_field_type",
                path=".".join(path),
                actual=type(value).__name__,
            )
    return scene_file


def restore_recorded_scene_base(
    *,
    scene: Any,
    raw_scene_file: Any,
    init_metadata: Mapping[str, Any],
    stopped_context: Callable[[], Any],
) -> Dict[str, Any]:
    """Restore the recorded clean scene and per-object metadata like official playback."""

    scene_file = normalize_recorded_scene_snapshot(raw_scene_file)
    if not isinstance(init_metadata, Mapping):
        raise TransitionManifestError(
            "recorded_init_metadata_invalid_type",
            actual=type(init_metadata).__name__,
        )

    # Deliberately do not call scene.reset(): its current _initial_file may come from a
    # different collector commit and is not the source serialization schema. Scene.restore
    # owns the single shared template-alias preflight for both generic and recorded paths.
    try:
        scene_restore_summary = scene.restore(scene_file, update_initial_file=True) or {}
    except SceneRestorePreflightError as exc:
        raise TransitionManifestError(exc.reason, **exc.details) from exc
    detached_template_aliases = scene_restore_summary.get("detached_system_template_aliases", [])
    with stopped_context():
        for attr, values in init_metadata.items():
            try:
                n_values = len(values)
            except TypeError as exc:
                raise TransitionManifestError(
                    "recorded_init_metadata_invalid_values",
                    attribute=attr,
                ) from exc
            if n_values != scene.n_objects:
                raise TransitionManifestError(
                    "recorded_init_metadata_object_count_mismatch",
                    attribute=attr,
                    metadata_count=n_values,
                    scene_object_count=scene.n_objects,
                )
        for index, obj in enumerate(scene.objects):
            for attr, values in init_metadata.items():
                value = values[index]
                setattr(obj, attr, value.item() if getattr(value, "ndim", None) == 0 else value)

    return {
        "recorded_scene_object_count": int(scene.n_objects),
        "recorded_init_metadata_keys": sorted(init_metadata.keys()),
        "detached_system_template_aliases": detached_template_aliases,
    }


def require_exact_cache_artifacts(cache: Mapping[str, Any]) -> None:
    """Fail closed unless a cache has the complete schema-v3 scene contract."""

    schema_version = int(cache.get("schema_version", 1))
    if schema_version != TRANSITION_STATE_CACHE_SCHEMA_VERSION:
        raise TransitionManifestError(
            "exact_cache_schema_v3_required",
            cache_schema_version=schema_version,
            required_schema_version=TRANSITION_STATE_CACHE_SCHEMA_VERSION,
        )
    if not cache.get("transition_manifest_present", False):
        raise TransitionManifestError("exact_cache_transition_manifest_missing")
    normalize_recorded_scene_snapshot(cache.get("recorded_scene_file_json"))
    if not isinstance(cache.get("init_metadata"), Mapping):
        raise TransitionManifestError("exact_cache_init_metadata_missing")


def _require_list(value: Any, *, step: int, path: str) -> list:
    if not isinstance(value, list):
        raise TransitionManifestError(
            "transition_manifest_invalid_field_type",
            step=step,
            path=path,
            expected="list",
            actual=type(value).__name__,
        )
    return value


def normalize_transition_manifest(raw_manifest: Any) -> Dict[int, Dict[str, Dict[str, list]]]:
    """Parse and strictly validate an HDF5 transition manifest.

    Args:
        raw_manifest: JSON text / bytes or an already-decoded mapping. ``None`` and
            empty text mean that no manifest is available and return an empty mapping.

    Returns:
        Mapping from non-negative simulator transition step to normalized event data.

    Raises:
        TransitionManifestError: If any field is malformed or constructor identity is
            absent. Exact restore must stop rather than guess in this case.
    """

    if raw_manifest is None:
        return {}
    if hasattr(raw_manifest, "item") and not isinstance(raw_manifest, (str, bytes, Mapping)):
        raw_manifest = raw_manifest.item()
    if isinstance(raw_manifest, bytes):
        raw_manifest = raw_manifest.decode("utf-8")
    if isinstance(raw_manifest, str):
        if not raw_manifest.strip():
            return {}
        try:
            raw_manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as exc:
            raise TransitionManifestError(
                "transition_manifest_invalid_json",
                line=exc.lineno,
                column=exc.colno,
            ) from exc
    if not isinstance(raw_manifest, Mapping):
        raise TransitionManifestError(
            "transition_manifest_invalid_root",
            expected="mapping",
            actual=type(raw_manifest).__name__,
        )

    normalized: Dict[int, Dict[str, Dict[str, list]]] = {}
    for raw_step, raw_event in raw_manifest.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise TransitionManifestError("transition_manifest_invalid_step", step=raw_step) from exc
        if step < 0 or str(step) != str(raw_step):
            raise TransitionManifestError("transition_manifest_invalid_step", step=raw_step)
        if step in normalized:
            raise TransitionManifestError("transition_manifest_duplicate_step", step=step)
        if not isinstance(raw_event, Mapping):
            raise TransitionManifestError(
                "transition_manifest_invalid_event",
                step=step,
                actual=type(raw_event).__name__,
            )

        event: Dict[str, Dict[str, list]] = {}
        for entity_kind in ("systems", "objects"):
            entity_event = raw_event.get(entity_kind)
            if not isinstance(entity_event, Mapping):
                raise TransitionManifestError(
                    "transition_manifest_missing_entity_section",
                    step=step,
                    entity_kind=entity_kind,
                )
            additions = _require_list(entity_event.get("add"), step=step, path=f"{entity_kind}.add")
            removals = _require_list(entity_event.get("remove"), step=step, path=f"{entity_kind}.remove")
            event[entity_kind] = {"add": additions, "remove": removals}

        for system_action in ("add", "remove"):
            values = event["systems"][system_action]
            if any(not isinstance(name, str) or not name for name in values):
                raise TransitionManifestError(
                    "transition_manifest_invalid_system_name",
                    step=step,
                    action=system_action,
                )
            if len(values) != len(set(values)):
                raise TransitionManifestError(
                    "transition_manifest_duplicate_system_name",
                    step=step,
                    action=system_action,
                )

        object_removals = event["objects"]["remove"]
        if any(not isinstance(name, str) or not name for name in object_removals):
            raise TransitionManifestError("transition_manifest_invalid_object_removal", step=step)
        if len(object_removals) != len(set(object_removals)):
            raise TransitionManifestError("transition_manifest_duplicate_object_removal", step=step)

        added_names = []
        for index, init_info in enumerate(event["objects"]["add"]):
            if not isinstance(init_info, Mapping):
                raise TransitionManifestError(
                    "transition_manifest_invalid_object_init_info",
                    step=step,
                    index=index,
                )
            class_module = init_info.get("class_module")
            class_name = init_info.get("class_name")
            args = init_info.get("args")
            if not isinstance(class_module, str) or not class_module:
                raise TransitionManifestError(
                    "transition_manifest_missing_object_class_module",
                    step=step,
                    index=index,
                )
            if not isinstance(class_name, str) or not class_name:
                raise TransitionManifestError(
                    "transition_manifest_missing_object_class_name",
                    step=step,
                    index=index,
                )
            if not isinstance(args, Mapping):
                raise TransitionManifestError(
                    "transition_manifest_missing_object_args",
                    step=step,
                    index=index,
                )
            name = args.get("name")
            if not isinstance(name, str) or not name:
                raise TransitionManifestError(
                    "transition_manifest_missing_object_name",
                    step=step,
                    index=index,
                )
            added_names.append(name)
        if len(added_names) != len(set(added_names)):
            raise TransitionManifestError("transition_manifest_duplicate_object_addition", step=step)

        normalized[step] = event

    return dict(sorted(normalized.items()))


def transition_events_before_state_frame(
    manifest: Mapping[int, Dict[str, Dict[str, list]]], state_frame: int
) -> Tuple[Tuple[int, Dict[str, Dict[str, list]]], ...]:
    """Return lifecycle events that precede a serialized state frame.

    DataPlaybackWrapper applies transition step ``i`` immediately before loading
    ``state[i + 1]``. Therefore state frame ``f`` requires exactly events with
    ``transition_step < f``; an event keyed by ``f`` must not be applied yet.
    """

    if int(state_frame) < 0:
        raise TransitionManifestError("invalid_target_state_frame", state_frame=state_frame)
    return tuple((step, event) for step, event in sorted(manifest.items()) if step < int(state_frame))


def materialize_transition_events(
    *,
    scene: Any,
    events: Sequence[Tuple[int, Dict[str, Dict[str, list]]]],
    create_object_from_init_info: Callable[[Mapping[str, Any]], Any],
    uuid_from_name: Callable[[str], int],
    park_object: Callable[[Any, int], None],
    initialize_event: Callable[[], None],
) -> Dict[str, Any]:
    """Replay validated lifecycle events against a clean scene.

    The operation follows DataPlaybackWrapper's authoritative order: systems add,
    systems remove, objects remove, objects add, then one initialization step. Every
    collision or missing removal fails closed. No UUID is assigned or substituted.
    """

    added_objects = []
    removed_objects = []
    added_systems = []
    removed_systems = []
    applied_steps = []

    for step, event in events:
        for system_name in event["systems"]["add"]:
            existing = scene.system_registry("name", system_name)
            if existing is not None:
                raise TransitionManifestError(
                    "transition_system_add_collision",
                    step=step,
                    system_name=system_name,
                )
            system = scene.get_system(system_name, force_init=True)
            registered = scene.system_registry("name", system_name)
            if registered is not system:
                raise TransitionManifestError(
                    "transition_system_registration_mismatch",
                    step=step,
                    system_name=system_name,
                )
            added_systems.append(system_name)

        for system_name in event["systems"]["remove"]:
            existing = scene.system_registry("name", system_name)
            if existing is None:
                raise TransitionManifestError(
                    "transition_system_removal_missing",
                    step=step,
                    system_name=system_name,
                )
            scene.clear_system(system_name)
            if scene.system_registry("name", system_name) is not None:
                raise TransitionManifestError(
                    "transition_system_removal_failed",
                    step=step,
                    system_name=system_name,
                )
            removed_systems.append(system_name)

        for object_name in event["objects"]["remove"]:
            obj = scene.object_registry("name", object_name)
            if obj is None:
                raise TransitionManifestError(
                    "transition_object_removal_missing",
                    step=step,
                    object_name=object_name,
                )
            scene.remove_object(obj)
            if scene.object_registry("name", object_name) is not None:
                raise TransitionManifestError(
                    "transition_object_removal_failed",
                    step=step,
                    object_name=object_name,
                )
            removed_objects.append(object_name)

        for init_info in event["objects"]["add"]:
            expected_name = init_info["args"]["name"]
            expected_uuid = int(uuid_from_name(expected_name))
            existing_by_name = scene.object_registry("name", expected_name)
            if existing_by_name is not None:
                raise TransitionManifestError(
                    "transition_object_name_collision",
                    step=step,
                    object_name=expected_name,
                    existing_type=type(existing_by_name).__name__,
                    expected_type=init_info["class_name"],
                )
            existing_by_uuid = scene.object_registry("uuid", expected_uuid)
            if existing_by_uuid is not None:
                raise TransitionManifestError(
                    "transition_object_uuid_collision",
                    step=step,
                    object_name=expected_name,
                    expected_uuid=expected_uuid,
                    existing_name=getattr(existing_by_uuid, "name", None),
                )

            obj = create_object_from_init_info(init_info)
            if obj.name != expected_name or int(obj.uuid) != expected_uuid:
                raise TransitionManifestError(
                    "transition_object_identity_mismatch",
                    step=step,
                    expected_name=expected_name,
                    actual_name=getattr(obj, "name", None),
                    expected_uuid=expected_uuid,
                    actual_uuid=getattr(obj, "uuid", None),
                )
            if type(obj).__module__ != init_info["class_module"] or type(obj).__name__ != init_info["class_name"]:
                raise TransitionManifestError(
                    "transition_object_type_mismatch",
                    step=step,
                    object_name=expected_name,
                    expected_type=f"{init_info['class_module']}.{init_info['class_name']}",
                    actual_type=f"{type(obj).__module__}.{type(obj).__name__}",
                )

            scene.add_object(obj)
            # Cumulative mid-frame reconstruction applies multiple transition events
            # without the intermediate state loads used by DataPlaybackWrapper. Use a
            # globally monotonic slot so additions from different events cannot overlap
            # at the same temporary parking position before initialization.
            global_addition_index = len(added_objects)
            park_object(obj, global_addition_index)
            if scene.object_registry("name", expected_name) is not obj:
                raise TransitionManifestError(
                    "transition_object_name_registration_mismatch",
                    step=step,
                    object_name=expected_name,
                )
            if scene.object_registry("uuid", expected_uuid) is not obj:
                raise TransitionManifestError(
                    "transition_object_uuid_registration_mismatch",
                    step=step,
                    object_name=expected_name,
                    expected_uuid=expected_uuid,
                )
            added_objects.append(expected_name)

        initialize_event()
        applied_steps.append(step)

    return {
        "applied_transition_steps": applied_steps,
        "added_objects": added_objects,
        "removed_objects": removed_objects,
        "added_systems": added_systems,
        "removed_systems": removed_systems,
    }
