"""Test: scripted pick-up bowl and drop it on the plate.

Uses the exact hardcoded pick actions from test_pick_bowl.py (proven to work),
then dynamically computes transport-to-plate trajectory from actual positions.
"""

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
env = PandaLiberoSpatial(task_id=0, config_overrides={"episode_length": 400})
rng = jax.random.PRNGKey(42)
state = env.reset(rng)

# Settle objects
mj_model = env.mj_model
mj_data = mjx.get_data(mj_model, state.data)
for _ in range(500):
    mujoco.mj_step(mj_model, mj_data)

# Zero residual velocities
for body_name in ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1", "cookies_1", "ramekin_1"]:
    bid = mj_model.body(body_name).id
    jnt_adr = mj_model.body_jntadr[bid]
    if jnt_adr >= 0:
        dof_adr = mj_model.jnt_dofadr[jnt_adr]
        mj_data.qvel[dof_adr : dof_adr + 6] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = state.replace(data=mjx.put_data(mj_model, mj_data, impl=env._config.impl))

# ── Read actual object positions after settling ───────────────────────────────
ROBOT_BASE = np.array([-0.66, 0.0, 0.912])
ACTION_SCALE = 0.005

bowl_world = np.array(mj_data.xpos[env._obj_body])
plate_world = np.array(mj_data.xpos[env._plate_body])

bowl_rf = bowl_world - ROBOT_BASE
plate_rf = plate_world - ROBOT_BASE

print(f"Bowl (robot frame):  [{bowl_rf[0]:.4f}, {bowl_rf[1]:.4f}, {bowl_rf[2]:.4f}]")
print(f"Plate (robot frame): [{plate_rf[0]:.4f}, {plate_rf[1]:.4f}, {plate_rf[2]:.4f}]")

# ── Pick phases: exact same actions as the working test_pick_bowl.py ──────────
# Phase 1: Diagonal approach — [1.0, -0.04, -0.77] for 27 steps
#   Displacement: [+0.135, -0.0054, -0.104] → tip arrives ~10mm behind bowl in x
steps1, action1 = 27, jp.array([1.0, -0.04, -0.77, 0.0, 0.0, 0.0, 1.0])

# Phase 2: Descend straight down — [0, 0, -1] for 28 steps
#   Displacement: [0, 0, -0.14] → tip at z≈0.01 (grasp height)
steps2, action2 = 28, jp.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])

# Phase 3: Close gripper — 30 steps
steps3, action3 = 30, jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

# Phase 4: Lift bowl — [0, 0, 1] for 40 steps (z≈0.01 → z≈0.21)
TRANSPORT_Z = 0.20
steps4, action4 = 40, jp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])

# ── Transport phases: dynamically computed from actual plate position ─────────
# After pick, tip is at approximately:
#   x ≈ 0.451 + 27*0.005*1.0 = 0.586
#   y ≈ 0.005 + 27*0.005*(-0.04) = -0.0004
#   z ≈ 0.01 + 40*0.005*1.0 = 0.21
tip_after_lift = np.array([0.586, -0.0004, 0.21])

# Phase 5: Move horizontally to above plate
delta5 = np.array([plate_rf[0] - tip_after_lift[0],
                    plate_rf[1] - tip_after_lift[1],
                    0.0])
steps5 = max(int(np.max(np.abs(delta5[:2])) / ACTION_SCALE) + 1, 1)
action5_xyz = delta5 / (steps5 * ACTION_SCALE)
action5 = jp.array([action5_xyz[0], action5_xyz[1], 0.0, 0.0, 0.0, 0.0, -1.0])

# Phase 6: Descend closer to plate before releasing (z=0.21 → z=0.10)
RELEASE_Z = 0.07
delta6_z = RELEASE_Z - TRANSPORT_Z
steps6 = int(abs(delta6_z) / ACTION_SCALE) + 1
action6_z = delta6_z / (steps6 * ACTION_SCALE)
action6 = jp.array([0.0, 0.0, action6_z, 0.0, 0.0, 0.0, -1.0])

# Phase 7: Release gripper
steps7, action7 = 10, jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# Phase 8: Wait for bowl to settle
steps8, action8 = 40, jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# Phase 9: Retreat up
steps9, action9 = 20, jp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

total_steps = steps1 + steps2 + steps3 + steps4 + steps5 + steps6 + steps7 + steps8 + steps9
print(f"\nTrajectory plan ({total_steps} total steps):")
print(f"  1. Diagonal approach:  {steps1} steps")
print(f"  2. Descend to bowl:    {steps2} steps")
print(f"  3. Close gripper:      {steps3} steps")
print(f"  4. Lift bowl:          {steps4} steps")
print(f"  5. Move to plate:      {steps5} steps, action_xy=[{action5_xyz[0]:.3f}, {action5_xyz[1]:.3f}]")
print(f"  6. Descend to plate:   {steps6} steps")
print(f"  7. Release:            {steps7} steps")
print(f"  8. Wait settle:        {steps8} steps")
print(f"  9. Retreat:            {steps9} steps")

phases = [
    (action1, steps1, "Diagonal approach"),
    (action2, steps2, "Descend to bowl"),
    (action3, steps3, "Close gripper"),
    (action4, steps4, "Lift bowl"),
    (action5, steps5, "Move to plate"),
    (action6, steps6, "Descend to plate"),
    (action7, steps7, "Release bowl"),
    (action8, steps8, "Wait settle"),
    (action9, steps9, "Retreat up"),
]

# ── Renderer setup ────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(mj_model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mj_model.camera("agentview").id

jit_step = jax.jit(env.step)
print("\nWarming up JIT...")
_ = jit_step(state, jp.zeros(7))

# ── Execute ───────────────────────────────────────────────────────────────────
frames = []
total_step = 0
bowl_bid = mj_model.body("akita_black_bowl_1").id
plate_bid = mj_model.body("plate_1").id

for action, n_steps, desc in phases:
    print(f"\n=== {desc} ({n_steps} steps) ===")
    for i in range(n_steps):
        state = jit_step(state, action)
        total_step += 1

        if total_step % 2 == 0:
            mj_data_render = mjx.get_data(mj_model, state.data)
            mujoco.mj_forward(mj_model, mj_data_render)
            renderer.update_scene(mj_data_render, cam)
            frames.append(renderer.render())

        if i % 10 == 0:
            pos = np.array(state.info["current_pos"])
            mj_tmp = mjx.get_data(mj_model, state.data)
            bowl_w = mj_tmp.xpos[bowl_bid]
            print(f"  step {total_step:3d}  tip: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]  bowl: [{bowl_w[0]:.4f}, {bowl_w[1]:.4f}, {bowl_w[2]:.4f}]")

# Final state
pos = np.array(state.info["current_pos"])
mj_data_final = mjx.get_data(mj_model, state.data)
bowl_pos = np.array(mj_data_final.xpos[bowl_bid])
plate_pos = np.array(mj_data_final.xpos[plate_bid])
print(f"\nFinal tip (robot frame):  [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
print(f"Final bowl (world):       [{bowl_pos[0]:.4f}, {bowl_pos[1]:.4f}, {bowl_pos[2]:.4f}]")
print(f"Plate (world):            [{plate_pos[0]:.4f}, {plate_pos[1]:.4f}, {plate_pos[2]:.4f}]")
print(f"Bowl-plate XY distance:   {np.linalg.norm(bowl_pos[:2] - plate_pos[:2]):.4f}")

renderer.close()
out_path = "renders/test_pick_and_place.mp4"
os.makedirs("renders", exist_ok=True)
mediapy.write_video(out_path, frames, fps=20)
print(f"\nSaved video to {out_path} ({len(frames)} frames)")
