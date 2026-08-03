import torch as th

import omnigibson.scenes.scene_base as scene_base
from omnigibson.scenes.scene_base import Scene


class _Registry:
    def __init__(self, objects=()):
        self.objects = list(objects)

    def __call__(self, key, value):
        for obj in self.objects:
            if getattr(obj, key) == value:
                return obj
        return None

    def get_dict(self, key):
        return {getattr(obj, key): obj for obj in self.objects}

    def remove(self, obj):
        self.objects.remove(obj)


class _PrimIdentity:
    def __init__(self):
        self.valid = True


class _Object:
    def __init__(self, name, prim_path, identity=None):
        self.name = name
        self.prim_path = prim_path
        self.identity = identity or _PrimIdentity()
        self.links = {}
        self.move_count = 0

    def set_position_orientation(self, position, orientation):
        if not self.identity.valid:
            raise RuntimeError(f"expired_prim_access:{self.prim_path}")
        self.move_count += 1


class _System:
    def __init__(self, name, particle_template, registered_alias):
        self.name = name
        self._particle_template = particle_template
        self.registered_alias = registered_alias


class _FakeSimulator:
    def __init__(self):
        self.batch_removed_names = []
        self.physics_steps = 0

    def is_stopped(self):
        return False

    def is_playing(self):
        return True

    def batch_remove_objects(self, objects):
        for obj in objects:
            obj.set_position_orientation(
                th.zeros(3),
                th.tensor([0.0, 0.0, 0.0, 1.0]),
            )
            obj.scene.object_registry.remove(obj)
            self.batch_removed_names.append(obj.name)

    def batch_add_objects(self, objects, scenes):
        if objects or scenes:
            raise AssertionError("test target scene does not add objects")

    def step_physics(self):
        self.physics_steps += 1


class _MinimalScene(Scene):
    def __init__(self, whole, initial_file):
        self._object_registry = _Registry([whole])
        self._system_registry = _Registry()
        self._initial_file = initial_file
        self._task_metadata = {}
        self.restore_summaries = []
        self.loaded_states = []
        whole.scene = self

    @property
    def object_registry(self):
        return self._object_registry

    @property
    def system_registry(self):
        return self._system_registry

    @property
    def objects(self):
        return self._object_registry.objects

    @property
    def n_objects(self):
        return len(self.objects)

    @property
    def active_systems(self):
        return {system.name: system for system in self._system_registry.objects}

    def clear_system(self, system_name):
        system = self.system_registry("name", system_name)
        # Clearing the owner invalidates the shared prim for both wrappers.
        system._particle_template.identity.valid = False
        self._system_registry.remove(system)

    def get_system(self, system_name, force_init=True):
        raise AssertionError("test target scene has no systems to add")

    def load_state(self, state, serialized=False):
        self.loaded_states.append(state)

    def restore(self, scene_file, update_initial_file=False):
        summary = super().restore(scene_file, update_initial_file=update_initial_file)
        self.restore_summaries.append(summary)
        return summary


def _scene_snapshot():
    return {
        "init_info": {"class_name": "_MinimalScene"},
        "state": {
            "registry": {
                "object_registry": {"whole": {}},
                "system_registry": {},
            }
        },
        "objects_info": {"init_info": {"whole": {}}},
    }


def test_generic_scene_reset_preflight_is_repeatable_after_transition_aliases(monkeypatch):
    fake_sim = _FakeSimulator()
    monkeypatch.setattr(scene_base.og, "sim", fake_sim)
    monkeypatch.setattr(scene_base, "recursively_convert_to_torch", lambda value: value)

    snapshot = _scene_snapshot()
    whole = _Object("whole", "/World/scene_0/whole")
    scene = _MinimalScene(whole=whole, initial_file=snapshot)

    for _ in range(2):
        identity = _PrimIdentity()
        prim_path = "/World/scene_0/diced__whole/template"
        internal_template = _Object("diced__whole_template", prim_path, identity=identity)
        registered_alias = _Object("diced__whole_template", prim_path, identity=identity)
        internal_template.scene = scene
        registered_alias.scene = scene
        system = _System("diced__whole", internal_template, registered_alias)
        scene.system_registry.objects.append(system)
        scene.object_registry.objects.append(registered_alias)

        # This is the production generic path: Scene.reset -> Scene.restore. The
        # preflight must detach the alias before clear_system invalidates its prim.
        scene.reset()

        assert identity.valid is False
        assert scene.object_registry("name", registered_alias.name) is None
        assert registered_alias.move_count == 0
        assert scene.restore_summaries[-1]["detached_system_template_aliases"] == [
            {
                "system_name": "diced__whole",
                "object_name": "diced__whole_template",
                "prim_path": prim_path,
                "released_material_prim_paths": [],
            }
        ]

    assert scene.object_registry.objects == [whole]
    assert fake_sim.batch_removed_names == []
    assert fake_sim.physics_steps == 2
