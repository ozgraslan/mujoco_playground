"""Test: scripted pick-up of the bowl from table center."""

import os
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import mediapy

from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import (
    PandaLiberoSpatial,
)

# ── Environment setup ─────────────────────────────────────────────────────────
env = PandaLiberoSpatial(task_id=0)
rng = jax.random.PRNGKey(42)
state = env.reset(rng)

# Settle objects
mj_model = env.mj_model
mj_data = mjx.get_data(mj_model, state.data)
for _ in range(500):
    mujoco.mj_step(mj_model, mj_data)

# Zero residual velocities on freejoint objects
for body_name in ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1", "cookies_1", "ramekin_1"]:
    bid = mj_model.body(body_name).id
    jnt_adr = mj_model.body_jntadr[bid]
    if jnt_adr >= 0:
        dof_adr = mj_model.jnt_dofadr[jnt_adr]
        mj_data.qvel[dof_adr : dof_adr + 6] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = state.replace(data=mjx.put_data(mj_model, mj_data, impl=env._config.impl))

# ── Renderer setup ────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(mj_model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mj_model.camera("agentview").id

jit_step = jax.jit(env.step)

# Warm up JIT with a dummy step
print("Warming up JIT...")
_ = jit_step(state, jp.zeros(7))

# ── Coordinate reference ──────────────────────────────────────────────────────
# Tip starts at roughly (0.451, 0.005, 0.254) in robot base frame.
# Bowl (table_center) in robot frame: ~(0.585, 0.0, 0.008).
# Bowl collision cylinder: radius=0.035, half-height=0.02, center offset z+0.01.
# Bowl top in robot frame: ~0.038.
# Grasp height (fingers around bowl mid-section): ~0.02 in robot frame.
#
# action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
# Each is multiplied by action_scale=0.005.
# gripper >= 0 → open, gripper < 0 → close.

# ── Scripted pick sequence ────────────────────────────────────────────────────
# We define phases as (action_vector, num_steps, description).
# Actions are normalized [-1, 1]; actual displacement = action * 0.005 per step.

phases = [
    # Phase 1: Move above the bowl — need +x ~0.134, ~-y 0.005, -z ~0.104
    # Move diagonally: mostly +x and -z, gripper open
    (jp.array([1.0, -0.04, -0.77, 0.0, 0.0, 0.0, 1.0]),  27, "Move above bowl"),

    # Phase 2: Descend to grasp height — go to z ~0.01 in robot frame
    # From ~0.15 need -0.14, so 28 steps at 0.005
    (jp.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]),      28, "Descend to bowl"),

    # Phase 3: Close gripper around bowl
    (jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),      30, "Close gripper"),

    # Phase 4: Lift up with bowl
    (jp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),      50, "Lift bowl"),
]

frames = []
total_step = 0

for action, n_steps, desc in phases:
    print(f"\n=== {desc} ({n_steps} steps) ===")
    for i in range(n_steps):
        state = jit_step(state, action)
        total_step += 1

        # Render every 2 steps
        if total_step % 2 == 0:
            mj_data_render = mjx.get_data(mj_model, state.data)
            mujoco.mj_forward(mj_model, mj_data_render)
            renderer.update_scene(mj_data_render, cam)
            frames.append(renderer.render())

        if i % 10 == 0:
            pos = np.array(state.info["current_pos"])
            print(f"  step {total_step:3d}  tip: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

# Print final state
pos = np.array(state.info["current_pos"])
mj_data_final = mjx.get_data(mj_model, state.data)
bowl_pos = mj_data_final.xpos[env._obj_body]
print(f"\nFinal tip pos (robot frame): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
print(f"Final bowl pos (world):      [{bowl_pos[0]:.4f}, {bowl_pos[1]:.4f}, {bowl_pos[2]:.4f}]")

renderer.close()

out_path = "renders/test_pick_bowl.mp4"
os.makedirs("renders", exist_ok=True)
mediapy.write_video(out_path, frames, fps=20)
print(f"\nSaved video to {out_path} ({len(frames)} frames)")
