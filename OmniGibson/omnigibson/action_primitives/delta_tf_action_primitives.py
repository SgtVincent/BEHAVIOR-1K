"""Delta-TF action primitives on the upstream cuRobo / controller stack.

No IK, trajectory optimizer, collision exporter or physics loop is implemented
here. The decoder accepts the BRVLA execution23 layout, NOT native joint23.
Saved pairwise deltas and model chunk-cumulative deltas are distinct inputs.
"""
import hashlib
import os
import pathlib
import tempfile
from enum import IntEnum

import gymnasium as gym
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    BaseActionPrimitiveSet,
)
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.controllers import JointController
from omnigibson.controllers.controller_view import ControllerView


class DeltaTFAction(IntEnum):
    MOVE = 0


#: Frozen replay continuous-gripper width range in metres (original producer
#: ``GRIPPER_RANGE_M``). Widths are metres, not finger joint qpos; the future
#: grasp-phase binding must normalise the width into the controller range.
GRIPPER_RANGE_M = 0.05


def _build_peer_curobo_profile(robot, chest_link):
    """Reuse the official DEFAULT cuRobo config and register the chest link as
    a planning kinematics link.

    The upstream yaml and curobo core are not edited. The merged profile is
    materialized in a per-run temp directory and handed to the official
    ``CuRoboMotionGenerator`` through its ``robot_cfg_path`` override. Every
    non-chest section is copied verbatim from the official yaml; the only
    mutation is appending ``chest_link`` to ``kinematics.link_names`` when it
    is absent.
    """
    official = robot.curobo_path
    if isinstance(official, dict):
        official_path = official.get(CuRoboEmbodimentSelection.DEFAULT)
    else:
        official_path = official
    if not official_path:
        raise ValueError("No official DEFAULT cuRobo profile path resolved")
    official_path = pathlib.Path(official_path)
    if not official_path.exists():
        raise ValueError(f"Official cuRobo profile absent: {official_path}")

    source_sha256 = hashlib.sha256(official_path.read_bytes()).hexdigest()
    config = yaml.safe_load(official_path.read_text(encoding="utf-8"))
    if config is None:
        raise ValueError(f"Official cuRobo profile empty: {official_path}")
    robot_cfg = config.get("robot_cfg", config)
    kinematics = robot_cfg["kinematics"]
    link_names = list(kinematics.get("link_names") or [])
    if chest_link not in link_names:
        link_names.append(chest_link)
    kinematics["link_names"] = link_names

    # Managed run directory only: B1K_PEER_PROFILE_DIR (runner-provided, under
    # /dev/shm lane or the run artifact dir), otherwise a TMPDIR-isolated temp
    # dir. Never fall back to an unrelated shared /tmp location; NAS is not
    # used for run-time assets.
    managed_dir = os.environ.get("B1K_PEER_PROFILE_DIR")
    if managed_dir:
        profile_dir = pathlib.Path(managed_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        dir_kind = "managed_env"
    else:
        profile_dir = pathlib.Path(
            tempfile.mkdtemp(prefix="b1k_curobo_peer_", dir=os.environ.get("TMPDIR") or None)
        )
        dir_kind = "tmpdir_isolated"
    peer_path = profile_dir / "r1pro_default_chest.yaml"
    peer_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    receipt = {
        "source_yaml": str(official_path),
        "source_sha256": source_sha256,
        "added_link": chest_link,
        "link_names": list(link_names),
        "peer_yaml": str(peer_path),
        "profile_dir": str(profile_dir),
        "profile_dir_kind": dir_kind,
    }
    return str(peer_path), receipt


#: Relative shear residual above which the world affine cannot be represented as
#: official-mesh-rotation . diag(scale). Above it, the mesh affine is baked into
#: robot-frame vertices instead of using a silent norm-scale approximation.
MESH_AFFINE_SHEAR_TOL = 1e-4


class _PeerCuRoboMotionGenerator(CuRoboMotionGenerator):
    """Parallel MotionGenerator adapter for the 2026 GeomPrim API.

    Upstream ``update_obstacles`` reads ``collision_mesh.get_world_scale()``,
    which the 2026 ``GeomPrim`` no longer exposes (trace 308: AttributeError at
    curobo.py:305). This subclass keeps the official obstacle-world / robot
    world flow and replaces only the per-mesh affine/scale construction.

    The full world affine (parent chain included) is read through the official
    ``usd_utils.get_world_pose_with_scale`` via ``XFormPrim.scaled_transform``.
    Against the official mesh-pose rotation ``R``, ``S = R.T @ L``: its diagonal
    is the signed per-axis scale (reflection sign kept on its original axis),
    and its off-diagonal is the shear residual. If the residual is within
    tolerance the diagonal-scale path is exact; otherwise the full affine is
    baked into robot-frame vertices with identity pose and unit scale. Column
    norms alone are never used as an approximate scale.

    No shared upstream file is edited, and there is no global monkey patch. The
    ``update_obstacles`` body mirrors upstream baseline ``bd049de``
    (curobo.py:265-311); only the mesh construction differs, so it can be
    followed/dropped when an upstream bugfix lands.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obstacle_report = []

    def _build_obstacle_mesh(self, collision_mesh, robot_transform):
        obj_pos, obj_quat = collision_mesh.get_position_orientation()
        obj_pose = T.pose2mat((obj_pos, obj_quat))
        pose = robot_transform @ obj_pose
        pos, orn = T.mat2pose(pose)
        orn = orn[[3, 0, 1, 2]]

        affine = th.as_tensor(
            collision_mesh.scaled_transform, dtype=th.float32
        ).detach().cpu()
        linear = affine[:3, :3]
        rot_world = T.quat2mat(obj_quat)
        decomposed = rot_world.T @ linear
        scale = th.diag(decomposed).clone()
        offdiag = decomposed - th.diag(th.diag(decomposed))
        shear = float(th.max(th.abs(offdiag)).item())
        ref_scale = max(1.0, float(th.max(th.abs(scale)).item()))
        points = collision_mesh.points.detach().cpu()
        faces = collision_mesh.faces.detach().cpu()

        meta = {
            "name": collision_mesh.prim_path,
            "shear_residual": shear,
            "shear_tolerance": MESH_AFFINE_SHEAR_TOL,
        }
        if shear <= MESH_AFFINE_SHEAR_TOL * ref_scale:
            meta["mode"] = "diagonal_scale"
            meta["scale"] = scale.tolist()
            vertices, mesh_pose, mesh_scale = points, th.cat([pos, orn]).tolist(), scale.numpy()
        else:
            # Bake the full official affine into robot-frame vertices; identity
            # pose, unit scale. Only the official world affine and robot
            # transform are used here.
            world_pts = T.transform_points(points, affine)
            world_h = th.cat([world_pts, th.ones((len(world_pts), 1))], dim=1)
            robot_pts = (robot_transform @ world_h.T).T[:, :3]
            if not th.isfinite(robot_pts).all():
                raise ValueError("Baked obstacle mesh vertices are non-finite")
            meta["mode"] = "baked_affine"
            meta["scale"] = [1.0, 1.0, 1.0]
            vertices = robot_pts
            mesh_pose = th.cat([th.zeros(3), th.tensor([1.0, 0.0, 0.0, 0.0])]).tolist()
            mesh_scale = th.ones(3).numpy()

        mesh = lazy.curobo.geom.types.Mesh(
            name=collision_mesh.prim_path,
            pose=mesh_pose,
            vertices=vertices.numpy(),
            faces=faces.numpy(),
            scale=mesh_scale,
        )
        self._obstacle_report.append(meta)
        return mesh

    def update_obstacles(self, ignore_objects=None):
        # Body mirrors upstream curobo.py:265-311 at baseline bd049de; follow
        # upstream when that method changes. Only the mesh constructor differs.
        obstacles = {"cuboid": None, "sphere": None, "mesh": [], "cylinder": None, "capsule": None}
        robot_transform = T.pose_inv(T.pose2mat(self.robot.root_link.get_position_orientation()))

        if og.sim.floor_plane is not None:
            prim = og.sim.floor_plane.prim.GetChildren()[0]
            m = lazy.curobo.util.usd_helper.get_mesh_attrs(
                prim, cache=self.usd_help._xform_cache, transform=robot_transform.numpy()
            )
            obstacles["mesh"].append(m)

        for obj in self.robot.scene.objects:
            if obj == self.robot:
                continue
            if obj.visual_only:
                continue
            if ignore_objects is not None and obj in ignore_objects:
                continue
            for link in obj.links.values():
                for collision_mesh in link.collision_meshes.values():
                    assert (
                        collision_mesh.geom_type == "Mesh"
                    ), f"collision_mesh {collision_mesh.prim_path} is not a mesh, but a {collision_mesh.geom_type}"
                    obstacles["mesh"].append(
                        self._build_obstacle_mesh(collision_mesh, robot_transform)
                    )

        world = lazy.curobo.geom.types.WorldConfig(**obstacles)
        world = world.get_collision_check_world()
        self.mg[CuRoboEmbodimentSelection.DEFAULT].update_world(world)


class DeltaTFDecoder:
    """Fixed-anchor execution23 -> absolute link poses, body displacement, widths.

    Layout: left pose6, right pose6, two gripper widths, base dx/dy/dyaw,
    chest pose6. Pose6 rotation is extrinsic xyz (Rz @ Ry @ Rx), metres/radians.
    The decoder owns no simulator or controller state. Failed consumers must not
    reuse a partially consumed decoder; reset/rebind an entire chunk explicitly.
    """

    def __init__(self, anchors, representation):
        if representation not in ("pairwise_delta", "chunk_cumulative"):
            raise ValueError("Specify pairwise_delta or chunk_cumulative explicitly")
        self.representation = representation
        self.anchors = {k: self._matrix(v) for k, v in anchors.items()}
        if set(self.anchors) != {"left", "right", "chest"}:
            raise ValueError("Exactly left, right and chest anchors are required")
        self.relative = {k: th.eye(4, dtype=th.float32) for k in self.anchors}
        self.previous_move = th.zeros(3, dtype=th.float32)

    @staticmethod
    def _matrix(pose):
        mat = th.as_tensor(pose, dtype=th.float32).detach().cpu().clone()
        if mat.shape != (4, 4) or not th.isfinite(mat).all():
            raise ValueError("Anchor must be a finite 4x4 SE3 matrix")
        r = mat[:3, :3]
        if not th.allclose(mat[3], th.tensor([0., 0., 0., 1.]), atol=1e-6):
            raise ValueError("Invalid homogeneous row")
        if not th.allclose(r.T @ r, th.eye(3), atol=1e-5) or not th.isclose(th.linalg.det(r), th.tensor(1.), atol=1e-5):
            raise ValueError("Invalid rotation matrix")
        return mat

    def decode(self, row):
        row = th.as_tensor(row, dtype=th.float32).detach().cpu()
        if row.shape != (23,) or not th.isfinite(row).all():
            raise ValueError("A finite execution23 row is required, not native joint actions")
        if th.any(row[12:14] < 0) or th.any(row[12:14] > GRIPPER_RANGE_M + 1e-6):
            raise ValueError(
                f"Gripper widths must be metres in [0, {GRIPPER_RANGE_M}]"
            )
        poses = {}
        for name, sl in (("left", slice(0, 6)), ("right", slice(6, 12)), ("chest", slice(17, 23))):
            p = row[sl]
            delta = T.pose2mat((p[:3], T.euler2quat(p[3:])))
            relative = self.relative[name] @ delta if self.representation == "pairwise_delta" else delta
            self.relative[name] = relative
            poses[name] = self.anchors[name] @ relative
        current = row[14:17].clone()
        if self.representation == "pairwise_delta":
            displacement = current
        else:
            diff = current - self.previous_move
            previous_rotation = T.euler2mat(th.tensor([0., 0., self.previous_move[2]]))
            translation = previous_rotation[:2, :2].T @ diff[:2]
            displacement = th.cat((translation, diff[2:3]))
            self.previous_move = current
        return {"poses": poses, "base_displacement": displacement, "gripper_widths": row[12:14].clone()}


class DeltaTFActionPrimitives(StarterSemanticActionPrimitives):
    """Parallel primitive set; reuse upstream planner and feedback execution.

    This is a waypoint-tracking adapter, not a time-faithful replay certificate.
    Timing / tracking of converted demonstrations remains a separate GPU gate.
    All requested tool links must exist in the upstream planner profile. Do not
    substitute source joints or silently drop unsupported chest/arm constraints.
    """

    def __init__(self, env, robot, tool_links, representation="pairwise_delta", batch_size=2):
        # Do not call Starter.__init__: it changes the global grasp window.
        BaseActionPrimitiveSet.__init__(self, env, robot)
        if robot.model != "r1pro" or set(tool_links) != {"left", "right", "chest"}:
            raise ValueError("R1Pro and explicit left/right/chest tool links are required")
        for name, (key, _) in robot.controllers.items():
            if not ControllerView.is_controller_type(key, JointController) or ControllerView.get_use_delta_commands(key):
                raise ValueError(f"{name} is incompatible with upstream q_to_action; use an explicit joint profile")
        self.tool_links = dict(tool_links)
        if len(set(self.tool_links.values())) != 3:
            raise ValueError("Tool links must be distinct")
        if any(link not in robot.links for link in self.tool_links.values()):
            raise ValueError("All tool links must be present in the live robot")
        self._enable_head_tracking = False
        peer_profile, profile_receipt = _build_peer_curobo_profile(robot, self.tool_links["chest"])
        official_paths = robot.curobo_path
        if isinstance(official_paths, dict):
            cfg_paths = dict(official_paths)
        else:
            cfg_paths = {CuRoboEmbodimentSelection.DEFAULT: official_paths}
        # The peer yaml overrides DEFAULT only; BASE/ARM keep their official
        # upstream profiles so the base channel can plan through the official
        # BASE embodiment as a separate sequential segment.
        cfg_paths[CuRoboEmbodimentSelection.DEFAULT] = peer_profile
        self._motion_generator = _PeerCuRoboMotionGenerator(
            robot,
            robot_cfg_path=cfg_paths,
            use_default_embodiment_only=False,
            batch_size=batch_size,
        )
        profile_receipt["embodiments"] = sorted(str(k) for k in cfg_paths.keys())
        profile_receipt["obstacle_meshes"] = self._motion_generator._obstacle_report
        self._plans_attempted = 0
        self._plans_succeeded = 0
        self._curobo_profile_receipt = profile_receipt
        self._emb = CuRoboEmbodimentSelection.DEFAULT
        if self._emb not in self._motion_generator.mg:
            available = sorted(str(k) for k in self._motion_generator.mg.keys())
            raise ValueError(
                f"Upstream cuRobo profile has no DEFAULT embodiment; available={available}. "
                "Bind a matching robot curobo profile; do not silently fall back to another embodiment."
            )
        supported = set(self._motion_generator.additional_links[self._emb]) | {self._motion_generator.ee_link[self._emb]}
        if not set(tool_links.values()) <= supported:
            raise ValueError("Upstream cuRobo profile lacks requested tool links; no silent constraint drop")
        self.representation = representation
        self.decoder = None

    def get_action_space(self):
        return gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(23,), dtype="float32")

    def bind_anchors(self):
        base = T.pose2mat(self.robot.links[self.robot.base_footprint_link_name].get_position_orientation())
        anchors = {
            name: T.pose_inv(base) @ T.pose2mat(self.robot.links[link].get_position_orientation())
            for name, link in self.tool_links.items()
        }
        self.decoder = DeltaTFDecoder(anchors, self.representation)

    def _navigate_base(self, displacement):
        """Plan and execute the holonomic base through the official BASE
        embodiment, mirroring upstream ``_navigate_to_pose``.

        This is a standalone base segment executed before the arm/chest plan;
        the two are not a synchronous full-body trajectory. The body-relative
        ``displacement`` (dx, dy, dyaw) is rotated into the world frame from
        the current base yaw. Raises if the BASE embodiment profile is absent
        or the planner rejects the target; the base channel is never dropped
        or threshold-loosened.
        """
        if CuRoboEmbodimentSelection.BASE not in self._motion_generator.mg:
            raise NotImplementedError(
                "Moving-base requires the official BASE cuRobo embodiment; absent in this profile"
            )
        base_pos, base_quat = self.robot.links[self.robot.base_footprint_link_name].get_position_orientation()
        yaw = T.quat2euler(base_quat)[2]
        cos, sin = th.cos(yaw), th.sin(yaw)
        dx, dy, dyaw = displacement
        world_x = (base_pos[0] + cos * dx - sin * dy).item()
        world_y = (base_pos[1] + sin * dx + cos * dy).item()
        world_yaw = (yaw + dyaw).item()

        pose_3d = self._get_robot_pose_from_2d_pose((world_x, world_y, world_yaw))
        target_pos = {self.robot.base_footprint_link_name: pose_3d[0]}
        target_quat = {self.robot.base_footprint_link_name: pose_3d[1]}
        target_pos = {
            k: th.stack([v for _ in range(self._motion_generator.batch_size)])
            for k, v in target_pos.items()
        }
        target_quat = {
            k: th.stack([v for _ in range(self._motion_generator.batch_size)])
            for k, v in target_quat.items()
        }
        self._plans_attempted += 1
        success, paths = self._motion_generator.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            is_local=False,
            ik_only=False,
            ik_world_collision_check=True,
            skip_obstacle_update=False,
            emb_sel=CuRoboEmbodimentSelection.BASE,
        )
        ok = th.where(success)[0]
        if len(ok) == 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR, "Official cuRobo BASE rejected the base target"
            )
        q = self._motion_generator.path_to_joint_trajectory(
            paths[int(ok[0])], get_full_js=True, emb_sel=CuRoboEmbodimentSelection.BASE
        ).cpu()
        if q.ndim != 2 or len(q) == 0 or not th.isfinite(q).all():
            raise ValueError("Official BASE planner returned an invalid trajectory")
        self._plans_succeeded += 1
        yield from self._execute_motion_plan(q, ignore_physics=False, ignore_failure=False)

    def apply(self, action):
        return self.apply_ref(DeltaTFAction.MOVE, action)

    def apply_ref(self, primitive, row):
        if primitive != DeltaTFAction.MOVE:
            raise ValueError("Unknown delta-TF primitive")
        if self.decoder is None:
            raise ValueError("Bind run/chunk anchors explicitly after the initial state is restored")
        decoded = self.decoder.decode(row)
        # Base motion plans through the official BASE embodiment as a separate
        # sequential segment BEFORE the arm/chest plan; it is not a synchronous
        # full-body plan. Original clock / frame0 / science gates are unchanged.
        base_disp = decoded["base_displacement"]
        if th.any(base_disp.abs() > 1e-7):
            yield from self._navigate_base(base_disp)
        for i, arm in enumerate(("left", "right")):
            measured = self.robot.get_joint_positions()[self.robot.gripper_control_idx[arm]]
            if not th.allclose(measured, decoded["gripper_widths"][i].expand_as(measured), atol=1e-4):
                raise NotImplementedError("Changing gripper width requires explicit collision-aware grasp-phase binding")
        base = T.pose2mat(self.robot.links[self.robot.base_footprint_link_name].get_position_orientation())
        targets = {self.tool_links[name]: T.mat2pose(base @ pose) for name, pose in decoded["poses"].items()}
        pos = {k: v[0].unsqueeze(0) for k, v in targets.items()}
        quat = {k: v[1].unsqueeze(0) for k, v in targets.items()}
        self._plans_attempted += 1
        success, paths = self._motion_generator.compute_trajectories(
            target_pos=pos, target_quat=quat, is_local=False, emb_sel=self._emb,
            ik_only=False, ik_world_collision_check=True, skip_obstacle_update=False,
        )
        ok = th.where(success)[0]
        if len(ok) == 0:
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR, "Official cuRobo rejected the target")
        q = self._motion_generator.path_to_joint_trajectory(paths[int(ok[0])], get_full_js=True, emb_sel=self._emb).cpu()
        if q.ndim != 2 or len(q) == 0 or not th.isfinite(q).all():
            raise ValueError("Official planner returned an invalid trajectory")
        current_q = self.robot.get_joint_positions().cpu()
        fixed_idx = th.cat((self.robot.base_control_idx, *[self.robot.gripper_control_idx[a] for a in ("left", "right")])).long()
        if not th.allclose(q[:, fixed_idx], current_q[fixed_idx].expand(len(q), -1), atol=1e-4):
            raise ValueError("Planner moved fixed base/gripper joints; this slice cannot execute that plan")
        self._plans_succeeded += 1
        yield from self._execute_motion_plan(q, ignore_physics=False, ignore_failure=False)
