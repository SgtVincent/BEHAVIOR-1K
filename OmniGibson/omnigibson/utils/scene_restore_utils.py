"""Fail-closed preflight helpers shared by all scene restore paths."""

from __future__ import annotations

from typing import Any, Dict, Mapping


class SceneRestorePreflightError(RuntimeError):
    """A scene restore cannot safely synchronize its current and target lifecycle."""

    def __init__(self, reason: str, **details: Any):
        self.reason = reason
        self.details = details
        detail_text = ", ".join(f"{key}={value!r}" for key, value in sorted(details.items()))
        super().__init__(reason if not detail_text else f"{reason}: {detail_text}")


def scene_restore_target_object_names(scene_info: Mapping[str, Any]) -> set[str]:
    """Return validated target names from constructor and serialized-state registries."""

    try:
        registries = {
            "objects_info.init_info": scene_info["objects_info"]["init_info"],
            "state.registry.object_registry": scene_info["state"]["registry"]["object_registry"],
        }
    except (KeyError, TypeError) as exc:
        raise SceneRestorePreflightError("scene_restore_target_object_registry_missing") from exc

    target_names = set()
    for source, registry in registries.items():
        if not isinstance(registry, Mapping):
            raise SceneRestorePreflightError(
                "scene_restore_target_object_registry_invalid",
                source=source,
                actual=type(registry).__name__,
            )
        for name in registry:
            if not isinstance(name, str) or not name:
                raise SceneRestorePreflightError(
                    "recorded_scene_snapshot_invalid_object_name",
                    source=source,
                    object_name=name,
                )
            target_names.add(name)
    return target_names


def _owned_xforms(obj: Any) -> list[Any]:
    """Return wrappers whose material-user bookkeeping is owned by an entity wrapper."""

    xforms = [obj]
    for link in (getattr(obj, "links", None) or {}).values():
        xforms.append(link)
        xforms.extend((getattr(link, "collision_meshes", None) or {}).values())
        xforms.extend((getattr(link, "visual_meshes", None) or {}).values())
    return xforms


def _plan_duplicate_alias_material_user_releases(*, alias: Any, template: Any) -> list[tuple[Any, Any]]:
    """Validate every alias-side material release without mutating runtime state."""

    template_xforms = set(_owned_xforms(template))
    releases = []
    for xform in _owned_xforms(alias):
        material = getattr(xform, "_material", None)
        if material is None:
            continue
        users = getattr(material, "users", None)
        if users is None or xform not in users:
            raise SceneRestorePreflightError(
                "system_template_alias_material_user_missing",
                object_name=getattr(alias, "name", None),
                material_prim_path=getattr(material, "prim_path", None),
            )
        if not any(user in template_xforms for user in users):
            raise SceneRestorePreflightError(
                "system_template_alias_material_not_shared_with_owner",
                object_name=getattr(alias, "name", None),
                material_prim_path=getattr(material, "prim_path", None),
            )
        releases.append((xform, material))
    return releases


def detach_system_template_aliases_for_restore(
    *,
    scene: Any,
    scene_info: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Detach exact duplicate registry aliases before a restore clears their owner system.

    Granular systems own an internal unregistered particle-template wrapper. Transition
    replay can register a distinct wrapper for the exact same name and prim. Clearing the
    system removes the shared prim, so the registered alias must leave the registry before
    the later object-diff removal. The complete plan is validated before any material user
    or registry is mutated.
    """

    target_names = scene_restore_target_object_names(scene_info)
    plans = []
    for system_name, system in scene.active_systems.items():
        template = getattr(system, "_particle_template", None)
        if template is None:
            continue
        template_name = getattr(template, "name", None)
        template_prim_path = getattr(template, "prim_path", None)
        if not isinstance(template_name, str) or not template_name:
            continue

        registered = scene.object_registry("name", template_name)
        if registered is None or registered is template:
            continue
        registered_prim_path = getattr(registered, "prim_path", None)
        if (
            not isinstance(template_prim_path, str)
            or registered_prim_path != template_prim_path
        ):
            raise SceneRestorePreflightError(
                "system_template_alias_identity_mismatch",
                system_name=system_name,
                object_name=template_name,
                system_template_prim_path=template_prim_path,
                registered_prim_path=registered_prim_path,
            )
        if template_name in target_names:
            raise SceneRestorePreflightError(
                "recorded_scene_target_contains_system_template_alias",
                system_name=system_name,
                object_name=template_name,
                prim_path=template_prim_path,
            )
        releases = _plan_duplicate_alias_material_user_releases(
            alias=registered,
            template=template,
        )
        plans.append(
            {
                "system_name": system_name,
                "object_name": template_name,
                "prim_path": template_prim_path,
                "registered": registered,
                "material_releases": releases,
            }
        )

    # Apply only after every alias and every material release has passed validation.
    detached = []
    for plan in plans:
        released_paths = []
        for xform, material in plan["material_releases"]:
            material.remove_user(xform)
            xform._material = None
            released_paths.append(getattr(material, "prim_path", None))
        scene.object_registry.remove(plan["registered"])
        detached.append(
            {
                "system_name": plan["system_name"],
                "object_name": plan["object_name"],
                "prim_path": plan["prim_path"],
                "released_material_prim_paths": released_paths,
            }
        )
    return detached
