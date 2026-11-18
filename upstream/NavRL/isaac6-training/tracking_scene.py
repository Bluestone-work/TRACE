"""Two rotor-driven vehicles, one physics clock, and a rendered RGB sensor."""

from pathlib import Path

import numpy as np

from navrl6_env import Isaac6NavEnv
from quadrotor_dynamics import QuadrotorActuator, rotation_matrix
from rotor_flight import RotorFlight
from visual_tracker import limit_speed


class FollowerVehicle(RotorFlight):
    # Same LiDAR sampling as the existing environment; only own shapes ignored.
    lidar_range = Isaac6NavEnv.lidar_range
    lidar_hbeams = Isaac6NavEnv.lidar_hbeams
    lidar_vbeams = Isaac6NavEnv.lidar_vbeams
    lidar_vfov = Isaac6NavEnv.lidar_vfov
    _raycast_lidar = Isaac6NavEnv._raycast_lidar

    def __init__(self, leader):
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics

        asset = Path(__file__).resolve().parents[1] / (
            "isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.usd")
        add_reference_to_stage(str(asset), "/World/Follower")
        self.world = leader.world
        self._query = leader._query
        self._self_query_ids = set()
        for prim in Usd.PrimRange(self.world.stage.GetPrimAtPath("/World/Follower")):
            self._self_query_ids.add(PhysicsSchemaTools.sdfPathToInt(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateDisableGravityAttr(False)
                api.CreateLinearDampingAttr(0.)
                api.CreateAngularDampingAttr(0.)
        self.drone = SingleArticulation("/World/Follower", name="follower")
        self.base_view = RigidPrim("/World/Follower/base_link", name="follower_body")
        self.rotor_view = RigidPrim("/World/Follower/rotor_[0-3]", name="follower_rotors")
        for obj in (self.drone, self.base_view, self.rotor_view):
            self.world.scene.add(obj)
        self.actuator = QuadrotorActuator(leader.physics_dt)

    def reset(self, position):
        self.rotor_joint_indices = np.array([self.drone.get_dof_index(f"rotor_{i}_joint")
                                             for i in range(4)])
        self.drone.set_world_pose(position=np.asarray(position), orientation=np.array([1., 0., 0., 0.]))
        self.drone.set_linear_velocity(np.zeros(3))
        self.drone.set_angular_velocity(np.zeros(3))
        self.actuator.reset()
        self.drone.set_joint_positions(np.zeros(4), joint_indices=self.rotor_joint_indices)
        self.drone.set_joint_velocities(self.actuator.rotor_speeds.astype(np.float32),
                                        joint_indices=self.rotor_joint_indices)
        self._read_physics_state()


def visual_tag(stage, path, color, radius):
    """A massless, noncolliding visual marker attached to the base link."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(0., 0., 0.25))
    material = UsdShade.Material.Define(stage, path + "Material")
    shader = UsdShade.Shader.Define(stage, path + "Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color) * 0.3)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
    return sphere


class TrackingScene:
    def __init__(self, seed=7, speed_limit=2., initial_distance=10., realistic=True,
                 scene_type="hall"):
        if scene_type not in ("hall", "forest"):
            raise ValueError(f"unknown scene_type: {scene_type}")
        self.scene_type = scene_type
        # The original single red obstacle is useful in the hall, while the
        # forest supplies its own collision geometry through tree trunks.
        self.leader = Isaac6NavEnv(dynamics="rotor", obstacle=scene_type != "forest")
        self.world = self.leader.world
        self.follower = FollowerVehicle(self.leader)
        self.speed_limit = speed_limit
        self.dt = self.leader.dt
        self.rng = np.random.default_rng(seed)
        if realistic and scene_type == "forest":
            from forest_scene import build_forest
            build_forest(self.leader, seed=seed + 100)
        elif realistic:
            from realistic_scene import build_hall
            build_hall(self.leader)
        else:
            from pxr import UsdLux
            UsdLux.DomeLight.Define(self.world.stage, "/World/TrackingLight").CreateIntensityAttr(1200.)
        self.marker = visual_tag(self.world.stage, "/World/Hummingbird/base_link/TrackingTag",
                                 (1., 0., 0.9), 0.18)
        visual_tag(self.world.stage, "/World/Follower/base_link/TrackingTag", (0., 0.55, 1.), 0.11)
        self.leader.reset()  # One world reset initializes BOTH articulations.
        self.initial_follower = np.array([-initial_distance * 0.92,
                                           -initial_distance * np.sqrt(1 - 0.92**2), 2.])
        self.follower.reset(self.initial_follower)
        self.world.render()
        self.leader._read_physics_state()
        self.steps = 0
        self.waypoints_reached = 0
        self.goal_changed = 0
        self.set_goal(np.array([9., -1., 2.]))
        self.goals = [self.leader.target.tolist()]
        self.assert_equal_dynamics()

    def assert_equal_dynamics(self):
        a, b = self.leader, self.follower
        for attr in ("mass", "inertia", "max_forces", "motor_alpha", "allocation"):
            np.testing.assert_allclose(getattr(a.actuator, attr), getattr(b.actuator, attr))
        np.testing.assert_allclose(a.base_view.get_masses(), b.base_view.get_masses())
        np.testing.assert_allclose(a.rotor_view.get_masses(), b.rotor_view.get_masses())
        print(f"TRACKING_EQUAL_DYNAMICS speed_limit={self.speed_limit} mass={a.actuator.mass}"
              f" initial_separation={np.linalg.norm(a.position - b.position):.3f}", flush=True)

    def set_goal(self, position):
        self.leader.target = np.asarray(position, dtype=np.float32)
        direction = self.leader.target - self.leader.position
        direction[2] = 0.
        self.leader.goal_direction = direction / max(np.linalg.norm(direction), 1e-6)
        self.goal_changed = self.steps

    def maybe_change_goal(self):
        reached = np.linalg.norm(self.leader.position - self.leader.target) < 0.65
        timeout = self.steps - self.goal_changed > 600
        if reached or timeout:
            self.waypoints_reached += int(reached)
            for _ in range(100):
                candidate = np.array([self.rng.uniform(-3., 12.), self.rng.uniform(-3.5, 3.5),
                                       self.rng.uniform(1.7, 2.7)])
                if (np.linalg.norm(candidate[:2] - self.leader.position[:2]) > 4.
                        and np.linalg.norm(candidate[:2] - np.array([2., 0.])) > 2.):
                    break
            self.set_goal(candidate)
            self.goals.append(candidate.tolist())
            print(f"TRACKING_A_NEW_GOAL reached={reached} goal={candidate.tolist()}", flush=True)

    def step(self, leader_command, follower_command):
        a_command = limit_speed(leader_command, self.speed_limit)
        b_command = limit_speed(follower_command, self.speed_limit)
        for _ in range(self.leader.physics_substeps):
            self.leader._apply_rotor_control(a_command)
            self.follower._apply_rotor_control(b_command)
            self.world.step(render=False)  # One shared 240 Hz clock, not two steps.
        self.leader._read_physics_state()
        self.follower._read_physics_state()
        self.steps += 1
        self.leader.step_count = self.steps
        return a_command, b_command

    def failures(self):
        result = []
        for name, drone in (("A", self.leader), ("B", self.follower)):
            tilt = np.arccos(np.clip(rotation_matrix(drone.orientation)[2, 2], -1., 1.))
            if not 0.3 < drone.position[2] < 5.5 or tilt > np.deg2rad(75.):
                result.append(name + "_unstable")
            lidar = drone._raycast_lidar()
            if (4. - lidar).min() < 0.24:
                result.append(name + "_collision_clearance")
        if np.linalg.norm(self.leader.position - self.follower.position) < 0.65:
            result.append("inter_vehicle_clearance")
        return result

    def close(self):
        self.leader.close()


class TrackingRGBCamera:
    def __init__(self, world, width=640, height=360):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self.world = world
        self.width, self.height = width, height
        self.path = "/World/B_Gimbal_RGB"
        self.horizontal_fov = 90.
        self._rep = rep
        rep.orchestrator.set_capture_on_play(False)
        camera = UsdGeom.Camera.Define(world.stage, self.path)
        camera.CreateFocalLengthAttr(18.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(36. * height / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.))
        self._transform = UsdGeom.Xformable(camera).AddTransformOp()
        self.render_product = rep.create.render_product(self.path, (width, height), name="B_RGB")
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.render_product])
        self.position = np.zeros(3)
        self.yaw = np.pi

    def set_pose(self, own_position, yaw):
        from pxr import Gf
        # Ideal stabilized yaw gimbal; pose depends ONLY on B's own state.
        # Keep the optical centre above B's identification sphere and rotors.
        self.position = np.asarray(own_position, dtype=np.float64) + np.array([0., 0., 0.50])
        self.yaw = float(yaw)
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.])
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*self.position),
                                        Gf.Vec3d(*(self.position + direction)), Gf.Vec3d(0, 0, 1))
        self._transform.Set(matrix.GetInverse())

    def read(self):
        data = self.annotator.get_data()
        if isinstance(data, dict):
            data = data.get("data")
        if data is None or np.asarray(data).size == 0:
            return None
        return np.asarray(data)[..., :3].copy()

    def capture(self):
        # Synchronous capture pairs current pixels with current gimbal pose.
        # delta_time=0 keeps rendering from advancing the physics timeline.
        before = self.world.current_time
        self._rep.orchestrator.step(delta_time=0.0, rt_subframes=2, pause_timeline=False)
        if abs(self.world.current_time - before) > 1e-7:
            raise RuntimeError("RGB capture unexpectedly advanced the physics clock")
        return self.read()

    def close(self):
        self.annotator.detach([self.render_product])
        self.render_product.destroy()
