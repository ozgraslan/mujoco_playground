# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LIBERO Spatial tasks: pick a black bowl from varying locations, place on plate."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics
from mujoco_playground._src.mjx_env import State


# Path to LIBERO assets relative to the workspace root.
_LIBERO_ASSETS_ROOT = (
    epath.Path(__file__).parent.parent.parent.parent.parent
    / "LIBERO"
    / "libero"
    / "libero"
    / "assets"
)

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────
# BDDL regions are (x_min, y_min, x_max, y_max) in LIBERO native coordinates.
# The MuJoCo scene uses LIBERO native coordinates directly (table at origin,
# surface z=0.9), so the mapping is identity for x/y.
# At reset time we sample a uniform 2D point within each region.

TABLE_SURFACE_Z = 0.9  # Table surface height


# ──────────────────────────────────────────────────────────────────────────────
# BDDL region definitions from libero_spatial tasks
# ──────────────────────────────────────────────────────────────────────────────
_BDDL_REGIONS = {
    "plate_region":                 (0.05, 0.19, 0.07, 0.21),
    # Moved further from plate (y +0.08)
    "next_to_plate_region":         (0.0, 0.3, 0.02, 0.32),
    "box_region":                   (0.06, 0.02, 0.08, 0.04),
    "next_to_box_region":           (0.12, -0.08, 0.14, -0.06),
    "between_plate_ramekin_region": (-0.06, 0.19, -0.04, 0.21),
    "ramekin_region":               (-0.21, 0.19, -0.19, 0.21),
    "next_to_ramekin_region":       (-0.19, 0.31, -0.17, 0.33),
    "table_center":                 (-0.1, -0.01, -0.05, 0.01),
    "cabinet_region":               (0.02, -0.28, 0.04, -0.26),
    "stove_region":                 (-0.31, -0.15, -0.29, -0.13),
}

# Distractor objects: (body_name, bddl_region_name, z_offset)
# bowl_2 region is now task-specific, see _BOWL2_REGION_BY_TASK below
_RANDOM_OBJECTS_BASE = [
    ("cookies_1",          "box_region",              0.02),
    ("ramekin_1",          "ramekin_region",          0.02),
    ("plate_1",            "plate_region",            0.025),
]

# Mapping from task_id to bowl_2 region (from BDDL)
_BOWL2_REGION_BY_TASK = {
    0: "next_to_plate_region",           # table_center task
    1: "next_to_ramekin_region",        # between_plate_ramekin_region task
    2: "next_to_ramekin_region",        # next_to_plate_region task
    3: "next_to_box_region",            # next_to_ramekin_region task
    4: "stove_region",                  # next_to_box_region task (flat_stove_1_cook_region)
    5: "cabinet_region",                # on cookie box (wooden_cabinet_1_top_side)
    6: "box_region",                    # on ramekin (cookies_1)
    7: "stove_region",                  # on wooden cabinet (flat_stove_1_cook_region)
    8: "cabinet_region",                # on stove (wooden_cabinet_1_top_side)
}

# ──────────────────────────────────────────────────────────────────────────────
# Spatial task definitions
# ──────────────────────────────────────────────────────────────────────────────
# Each task specifies: (description, bddl_region_name, z_offset)
# For "on_X" tasks the z_offset places the bowl on top of the other object.

_BOWL_Z_OFFSETS = {
    "table_center": 0.02,
    "between_plate_ramekin_region": 0.02,
    "next_to_plate_region": 0.02,
    "next_to_box_region": 0.02,
    "ramekin_region": 0.048,
    "next_to_ramekin_region": 0.00,
    "cabinet_region": 0.22, 
    "stove_region": 0.03,
    "box_region": 0.04}
_SPATIAL_TASKS = {
    0: (
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "table_center",
    ),
    1: (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "between_plate_ramekin_region",
    ),
    2: (
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "next_to_plate_region",
    ),
    3: (
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "next_to_ramekin_region",
    ),
    4: (
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "next_to_box_region",
    ),
    5: (
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "box_region",
    ),
    6: (
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "ramekin_region",
    ),
    7: (
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
        "cabinet_region",
    ),
    8: (
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "stove_region",
    ),
}


def _euler_to_rot(euler: jax.Array) -> jax.Array:
  """Convert RPY Euler angles to a 3x3 rotation matrix (ZYX convention)."""
  roll, pitch, yaw = euler[0], euler[1], euler[2]
  cr, sr = jp.cos(roll), jp.sin(roll)
  cp, sp = jp.cos(pitch), jp.sin(pitch)
  cy, sy = jp.cos(yaw), jp.sin(yaw)
  return jp.array([
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr],
  ])


def _get_libero_assets() -> Dict[str, bytes]:
  """Load LIBERO mesh and texture assets."""
  assets = {}
  libero_root = _LIBERO_ASSETS_ROOT

  # Akita black bowl — unique basename: bowl_texture.png, akita_black_bowl_vis.msh
  bowl_dir = libero_root / "stable_scanned_objects" / "akita_black_bowl"
  assets["bowl_texture.png"] = (bowl_dir / "texture.png").read_bytes()
  assets["akita_black_bowl_vis.msh"] = (
      bowl_dir / "visual" / "akita_black_bowl_vis.msh"
  ).read_bytes()

  # Plate — unique basename: plate_texture.png, plate_vis.msh
  plate_dir = libero_root / "stable_scanned_objects" / "plate"
  assets["plate_texture.png"] = (plate_dir / "texture.png").read_bytes()
  assets["plate_vis.msh"] = (
      plate_dir / "visual" / "model_vis.msh"
  ).read_bytes()

  # Cookies — unique basename: cookies_texture_map.png, cookies_vis.msh
  cookies_dir = libero_root / "stable_hope_objects" / "cookies"
  assets["cookies_texture_map.png"] = (cookies_dir / "texture_map.png").read_bytes()
  assets["cookies_vis.msh"] = (
      cookies_dir / "visual" / "cookies_vis.msh"
  ).read_bytes()

  # Ramekin — unique basename: ramekin_texture.png, ramekin_vis.msh
  ramekin_dir = libero_root / "stable_scanned_objects" / "glazed_rim_porcelain_ramekin"
  assets["ramekin_texture.png"] = (ramekin_dir / "texture.png").read_bytes()
  assets["ramekin_vis.msh"] = (
      ramekin_dir / "visual" / "glazed_rim_porcelain_ramekin_vis.msh"
  ).read_bytes()

  # Wooden cabinet — unique basenames with cabinet_ prefix
  cabinet_dir = libero_root / "articulated_objects" / "wooden_cabinet"
  assets["cabinet_dark_fine_wood.png"] = (
      cabinet_dir / "dark_fine_wood.png"
  ).read_bytes()
  assets["cabinet_metal.png"] = (cabinet_dir / "metal.png").read_bytes()
  for part in [
      "wooden_cabinet_base",
      "wooden_cabinet_top",
      "wooden_cabinet_top_handle",
      "wooden_cabinet_middle",
      "wooden_cabinet_middle_handle",
      "wooden_cabinet_bottom",
      "wooden_cabinet_bottom_handle",
  ]:
    assets[f"{part}_vis.msh"] = (
        cabinet_dir / part / "visual" / f"{part}_vis.msh"
    ).read_bytes()

  # Flat stove — unique basenames with stove_ prefix
  stove_dir = libero_root / "articulated_objects" / "flat_stove"
  assets["stove_metal.png"] = (stove_dir / "metal.png").read_bytes()
  assets["stove_knob_button_vis.msh"] = (
      stove_dir / "stove_knob_button" / "visual" / "stove_knob_button_vis.msh"
  ).read_bytes()
  assets["burnerplate.stl"] = (
      stove_dir / "stove_burner" / "burnerplate.stl"
  ).read_bytes()

  # LIBERO scene textures
  textures_dir = libero_root / "textures"
  assets["libero_floor_tile.png"] = (
      textures_dir / "tile_grigia_caldera_porcelain_floor.png"
  ).read_bytes()
  assets["libero_wood_table.png"] = (
      textures_dir / "martin_novak_wood_table.png"
  ).read_bytes()
  assets["libero_wood_table_legs.png"] = (
      textures_dir / "martin_novak_wood_table.png"
  ).read_bytes()
  assets["libero_wall_plaster.png"] = (
      textures_dir / "smooth_light_gray_plaster.png"
  ).read_bytes()

  return assets


def default_config() -> config_dict.ConfigDict:
  config = config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=500,
      action_repeat=1,
      action_scale=0.005,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_bowl=4.0,
              bowl_plate=8.0,
              no_table_collision=0.25,
              robot_target_qpos=0.3,
          ),
          no_soln_reward=-0.01,
      ),
      impl='warp',
      naconmax=24 * 2048,
      naccdmax=24 * 2048,
      njmax=2048,
  )
  return config


class PandaLiberoSpatial(mjx_env.MjxEnv):
  """LIBERO Spatial: pick a black bowl from a spatial location, place on plate.

  The task_id (0-8) selects the bowl starting position from the 9 supported
  LIBERO spatial task variants (all table-surface tasks; excludes the drawer
  task).
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      task_id: int = 0,
  ):
    if task_id not in _SPATIAL_TASKS:
      raise ValueError(
          f"task_id must be 0-8, got {task_id}. "
          f"Available: {list(_SPATIAL_TASKS.keys())}"
      )
    self._task_id = task_id
    self._task_description, bowl_region_name = _SPATIAL_TASKS[task_id]
    z_offset = _BOWL_Z_OFFSETS[bowl_region_name]

    # Compute bowl BDDL region bounds for sampling
    region = _BDDL_REGIONS[bowl_region_name]
    self._bowl_region_lo = jp.array([region[0], region[1]], dtype=jp.float32)
    self._bowl_region_hi = jp.array([region[2], region[3]], dtype=jp.float32)
    self._bowl_z = jp.float32(TABLE_SURFACE_Z + z_offset)

    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_libero_spatial.xml"
    )

    super().__init__(config, config_overrides)

    # Build combined assets: Panda menagerie + scene XMLs + LIBERO meshes
    self._xml_path = xml_path.as_posix()
    xml = xml_path.read_text()

    self._model_assets = panda.get_assets()
    self._model_assets.update(_get_libero_assets())

    mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)

    # Place the Panda robot base to match LIBERO: (-0.66, 0, 0.912)
    mj_model.body('link0').pos[:] = [-0.6, 0.0, 0.912]
    mj_model.opt.timestep = self.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
    self._action_scale = config.action_scale

    self._post_init(obj_name="akita_black_bowl_1", keyframe="home")

    # Use FK to initialise Cartesian control
    self._start_tip_transform = panda_kinematics.compute_franka_fk(
        self._init_ctrl[:7]
    )

    # Plate body (target location)
    self._plate_body = self._mj_model.body("plate_1").id

    # Track qposadr and region bounds for distractor objects.
    # Add bowl_2 with task-specific region
    self._rand_obj_qposadr = []
    self._rand_obj_region_lo = []
    self._rand_obj_region_hi = []
    self._rand_obj_z = []

    # Add bowl_2 with task-specific region
    bowl2_region = _BOWL2_REGION_BY_TASK[self._task_id]
    bowl2_z = _BOWL_Z_OFFSETS[bowl2_region]
    body = self._mj_model.body("akita_black_bowl_2")
    qposadr = self._mj_model.jnt_qposadr[body.jntadr[0]]
    region = _BDDL_REGIONS[bowl2_region]
    self._rand_obj_qposadr.append(qposadr)
    self._rand_obj_region_lo.append(jp.array([region[0], region[1]], dtype=jp.float32))
    self._rand_obj_region_hi.append(jp.array([region[2], region[3]], dtype=jp.float32))
    self._rand_obj_z.append(jp.float32(TABLE_SURFACE_Z + bowl2_z))

    # Add other distractors
    for name, dist_region_name, obj_z_offset in _RANDOM_OBJECTS_BASE:
        body = self._mj_model.body(name)
        qposadr = self._mj_model.jnt_qposadr[body.jntadr[0]]
        region = _BDDL_REGIONS[dist_region_name]
        self._rand_obj_qposadr.append(qposadr)
        self._rand_obj_region_lo.append(jp.array([region[0], region[1]], dtype=jp.float32))
        self._rand_obj_region_hi.append(jp.array([region[2], region[3]], dtype=jp.float32))
        self._rand_obj_z.append(jp.float32(TABLE_SURFACE_Z + obj_z_offset))

    # Contact sensor IDs (table collision)
    self._table_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_table_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def _post_init(self, obj_name: str, keyframe: str):
    """Initialise robot joint indices, sensors, and keyframe data."""
    arm_joints = [
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7",
    ]
    finger_joints = ["finger_joint1", "finger_joint2"]
    all_joints = arm_joints + finger_joints
    self._robot_arm_qposadr = jp.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in arm_joints
    ])
    self._robot_qposadr = jp.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.site("gripper").id
    self._obj_body = self._mj_model.body(obj_name).id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body(obj_name).jntadr[0]
    ]
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  @property
  def task_id(self) -> int:
    return self._task_id

  @property
  def task_description(self) -> str:
    return self._task_description

  def reset(self, rng: jax.Array) -> State:
    rng, rng_bowl = jax.random.split(rng)

    # Bowl position: sample uniformly within BDDL region
    bowl_xy = self._bowl_region_lo + jax.random.uniform(rng_bowl, (2,)) * (
        self._bowl_region_hi - self._bowl_region_lo
    )
    bowl_pos = jp.array([bowl_xy[0], bowl_xy[1], self._bowl_z])

    # Set initial qpos with bowl at the computed position
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(bowl_pos)
    )

    # Random yaw for bowl (quat stored at qposadr+3)
    rng, rng_yaw = jax.random.split(rng)
    yaw = jax.random.uniform(rng_yaw, (), minval=-jp.pi, maxval=jp.pi)
    quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    init_q = init_q.at[self._obj_qposadr + 3 : self._obj_qposadr + 7].set(quat)

    # Randomize distractor object positions and yaw rotations
    for qposadr, lo, hi, z in zip(
        self._rand_obj_qposadr,
        self._rand_obj_region_lo,
        self._rand_obj_region_hi,
        self._rand_obj_z,
    ):
      rng, rng_obj, rng_yaw = jax.random.split(rng, 3)
      obj_xy = lo + jax.random.uniform(rng_obj, (2,)) * (hi - lo)
      obj_pos = jp.array([obj_xy[0], obj_xy[1], z])
      init_q = init_q.at[qposadr : qposadr + 3].set(obj_pos)
      # Random yaw rotation
      yaw = jax.random.uniform(rng_yaw, (), minval=-jp.pi, maxval=jp.pi)
      quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
      init_q = init_q.at[qposadr + 3 : qposadr + 7].set(quat)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )

    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {
        "rng": rng,
        "reached_bowl": jp.array(0.0, dtype=float),
        "current_pos": self._start_tip_transform[:3, 3],
        "current_rot": self._start_tip_transform[:3, :3],
        "_steps": jp.array(0, dtype=int),
    }
    obs = self._get_obs(data, info)
    obs = jp.concat([obs, jp.zeros(1), jp.zeros(self.action_size)], axis=0)
    reward, done = jp.zeros(2)
    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    newly_reset = state.info['_steps'] == 0
    state.info['current_pos'] = jp.where(
        newly_reset, self._start_tip_transform[:3, 3], state.info['current_pos']
    )
    state.info['current_rot'] = jp.where(
        newly_reset, self._start_tip_transform[:3, :3], state.info['current_rot']
    )

    # Cartesian IK control: action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
    ctrl, new_tip_pos, new_tip_rot, no_soln = self._move_tip(
        state.info['current_pos'],
        state.info['current_rot'],
        state.data.ctrl,
        action,
    )
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    state.info['current_pos'] = new_tip_pos
    state.info['current_rot'] = new_tip_rot

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    reward += no_soln * self._config.reward_config.no_soln_reward

    bowl_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(bowl_pos[:2]) > 1.5)
    out_of_bounds |= bowl_pos[2] < 0.7
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    state.info['_steps'] += self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )

    obs = self._get_obs(data, state.info)
    obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
    return State(data, obs, reward, done.astype(float), state.metrics, state.info)

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    bowl_pos = data.xpos[self._obj_body]
    plate_pos = data.xpos[self._plate_body]
    gripper_pos = data.site_xpos[self._gripper_site]

    # Gripper → bowl distance reward
    gripper_bowl_dist = jp.linalg.norm(bowl_pos - gripper_pos)
    gripper_bowl = 1 - jp.tanh(5 * gripper_bowl_dist)

    # Bowl → plate distance reward (activated once the gripper has reached the bowl)
    bowl_plate_dist = jp.linalg.norm(plate_pos - bowl_pos)
    bowl_plate = 1 - jp.tanh(5 * bowl_plate_dist)

    # Track whether gripper has reached the bowl
    info["reached_bowl"] = 1.0 * jp.maximum(
        info["reached_bowl"],
        (gripper_bowl_dist < 0.012),
    )

    # Regularize arm toward home configuration
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Table collision penalty
    hand_table_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._table_hand_found_sensor
    ]
    table_collision = sum(hand_table_collision) > 0
    no_table_collision = (1 - table_collision).astype(float)

    return {
        "gripper_bowl": gripper_bowl,
        "bowl_plate": bowl_plate * info["reached_bowl"],
        "no_table_collision": no_table_collision,
        "robot_target_qpos": robot_target_qpos,
    }

  def _move_tip(
      self,
      current_tip_pos: jax.Array,
      current_tip_rot: jax.Array,
      current_ctrl: jax.Array,
      action: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Calculate new tip pose from Cartesian increment.

    Args:
      current_tip_pos: (3,) current EE position in robot base frame.
      current_tip_rot: (3,3) current EE rotation matrix.
      current_ctrl:    (8,) current joint ctrl.
      action:          (7,) [dx, dy, dz, droll, dpitch, dyaw, gripper].

    Returns:
      (new_ctrl, new_tip_pos, new_tip_rot, no_soln)
    """
    # Position update
    scaled_pos = action[:3] * self._config.action_scale
    new_tip_pos = current_tip_pos + scaled_pos

    # Clip to workspace bounds (robot base frame)
    new_tip_pos = new_tip_pos.at[0].set(jp.clip(new_tip_pos[0], 0.2, 1.0))
    new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], -0.5, 0.5))
    new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], -0.15, 0.5))

    # Orientation update
    scaled_rot = action[3:6] * self._config.action_scale
    delta_rot = _euler_to_rot(scaled_rot)
    new_tip_rot = delta_rot @ current_tip_rot

    # Build 4x4 transform for IK
    new_tip_mat = jp.identity(4)
    new_tip_mat = new_tip_mat.at[:3, :3].set(new_tip_rot)
    new_tip_mat = new_tip_mat.at[:3, 3].set(new_tip_pos)

    current_ctrl = jp.asarray(current_ctrl) ## check this later
    out_jp = panda_kinematics.compute_franka_ik(
        new_tip_mat, current_ctrl[6], current_ctrl[:7]
    )
    no_soln = jp.any(jp.isnan(out_jp))
    out_jp = jp.where(no_soln, current_ctrl[:7], out_jp)
    no_soln = jp.logical_or(no_soln, jp.any(jp.isnan(out_jp)))
    new_tip_pos = jp.where(no_soln, current_tip_pos, new_tip_pos)
    new_tip_rot = jp.where(no_soln, current_tip_rot, new_tip_rot)

    new_ctrl = current_ctrl.at[:7].set(out_jp)
    # Continuous gripper: action[6] in [-1, 1] maps proportionally to delta
    claw_delta = action[6] * 0.02
    new_ctrl = new_ctrl.at[7].set(new_ctrl[7] + claw_delta)

    return new_ctrl, new_tip_pos, new_tip_rot, no_soln

  @property
  def action_size(self) -> int:
    return 7

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    bowl_pos = data.xpos[self._obj_body]
    bowl_mat = data.xmat[self._obj_body].ravel()
    plate_pos = data.xpos[self._plate_body]

    return jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],  # partial rotation matrix (6 values)
        bowl_mat[3:],     # partial rotation matrix (6 values)
        bowl_pos - gripper_pos,    # relative: gripper → bowl
        plate_pos - bowl_pos,      # relative: bowl → plate
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])
