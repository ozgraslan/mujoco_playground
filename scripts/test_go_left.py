"""Test IK controller by commanding the robot to move left, saving a video."""

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

# Create env and reset
env = PandaLiberoSpatial(task_id=0)
rng = jax.random.PRNGKey(42)
state = env.reset(rng)

# Settle objects via mj_step
mj_model = env.mj_model
mj_data = mjx.get_data(mj_model, state.data)
for _ in range(500):
    mujoco.mj_step(mj_model, mj_data)

# Zero out velocities of freejoint objects after settling
for body_name in ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1", "cookies_1", "ramekin_1"]:
    bid = mj_model.body(body_name).id
    jnt_adr = mj_model.body_jntadr[bid]
    if jnt_adr >= 0:
        dof_adr = mj_model.jnt_dofadr[jnt_adr]
        mj_data.qvel[dof_adr : dof_adr + 6] = 0.0
mujoco.mj_forward(mj_model, mj_data)

# Put settled data back into mjx state
state = state.replace(data=mjx.put_data(mj_model, mj_data, impl=env._config.impl))

# Renderer
renderer = mujoco.Renderer(mj_model, height=480, width=640)
cam = mujoco.MjvCamera()
cam_id = mj_model.camera("agentview").id
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = cam_id

# Action: move left (+y), no rotation, gripper open
action_left = jp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

frames = []
n_steps = 100

# JIT compile step
jit_step = jax.jit(env.step)

print("Warming up JIT...")
_ = jit_step(state, action_left)

print(f"Running {n_steps} steps with left action...")
for i in range(n_steps):
    state = jit_step(state, action_left)

    # Render every 2 steps
    if i % 2 == 0:
        mj_data = mjx.get_data(mj_model, state.data)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, cam)
        frame = renderer.render()
        frames.append(frame)

    if i % 20 == 0:
        pos = np.array(state.info["current_pos"])
        print(f"  step {i:3d}  tip_pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

renderer.close()

out_path = "renders/test_go_left.mp4"
os.makedirs("renders", exist_ok=True)
mediapy.write_video(out_path, frames, fps=20)
print(f"Saved video to {out_path} ({len(frames)} frames)")
