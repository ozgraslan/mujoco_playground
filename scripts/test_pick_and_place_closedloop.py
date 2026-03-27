"""Closed-loop pick-and-place: dynamically computes actions from live positions."""

import os
from xml.parsers.expat import model
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import mediapy

from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import (
    PandaLiberoSpatial,
    default_config,
)

ROBOT_BASE = np.array([-0.6, 0.0, 0.912])
ACTION_SCALE = 0.005
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# Tolerances
XY_TOL = 0.008       # "close enough" in XY
Z_TOL = 0.005        # "close enough" in Z
GRASP_STEPS = 30     # steps to hold gripper closed
RELEASE_STEPS = 10   # steps to hold gripper open
SETTLE_STEPS = 40    # steps to let bowl settle after release

# Bowl geometry (from collision geoms: wall boxes at ~0.033-0.034 from center)
BOWL_HALF_DIAMETER = 0.034  # half diameter of the bowl

# Heights (robot base frame)
APPROACH_Z = 0.15    # hover height before descending
GRASP_Z = 0.01       # grasp height (proven to work in open-loop test)
LIFT_Z = 0.20        # lift height after grasping
RELEASE_Z = 0.07     # height above plate to release


def get_bowl_rf(state, env):
    """Bowl position in robot base frame."""
    return np.array(state.data.xpos[env._obj_body]) - ROBOT_BASE


def get_plate_rf(state, env):
    """Plate position in robot base frame."""
    return np.array(state.data.xpos[env._plate_body]) - ROBOT_BASE


def get_tip(state):
    """EE tip position in robot base frame."""
    return np.array(state.info["current_pos"])


# ── PD Controller ─────────────────────────────────────────────────────────────
# Action space: [-1, 1], each unit = ACTION_SCALE = 0.005m per step
# So to move 0.005m in one step, action = 1.0
# Kp: error in meters → action units. Kp=200 means 0.005m error → action=1.0
KP = 150.0   # proportional gain
KD = 15.0    # derivative gain (damps oscillation)

prev_error = np.zeros(3)

def pd_move(current, target, gripper, dt=1.0):
    """PD controller: compute action to move tip toward target."""
    global prev_error
    error = target - current
    derror = (error - prev_error) / dt
    prev_error = error.copy()

    # PD output → action units
    action_xyz = KP * error + KD * derror
    # Clip to [-1, 1]
    action_xyz = np.clip(action_xyz, -1.0, 1.0)
    return jp.array([action_xyz[0], action_xyz[1], action_xyz[2], 0.0, 0.0, 0.0, gripper])


def at_target(current, target, xy_tol=XY_TOL, z_tol=Z_TOL):
    """Check if tip is close enough to target."""
    return np.linalg.norm(current[:2] - target[:2]) < xy_tol and abs(current[2] - target[2]) < z_tol


# ── PID Controller ─────────────────────────────────────────────────────────────
# Add integral term for smoother, more accurate control
KI = 5.0     # integral gain (should be much smaller than KP)
INTEGRAL_WINDUP_LIMIT = 0.05  # max abs value for each integral component (meters)

prev_error = np.zeros(3)
error_integral = np.zeros(3)

def pd_move(current, target, gripper, dt=1.0):
    """PID controller: compute action to move tip toward target."""
    global prev_error, error_integral
    error = target - current
    derror = (error - prev_error) / dt
    error_integral += error * dt
    # Anti-windup: clip integral
    error_integral = np.clip(error_integral, -INTEGRAL_WINDUP_LIMIT, INTEGRAL_WINDUP_LIMIT)
    prev_error = error.copy()

    # PID output → action units
    action_xyz = KP * error + KD * derror + KI * error_integral
    # Clip to [-1, 1]
    action_xyz = np.clip(action_xyz, -1.0, 1.0)
    return jp.array([action_xyz[0], action_xyz[1], action_xyz[2], 0.0, 0.0, 0.0, gripper])
dt = 0.05
config = default_config()
for task_id in [7]:
    # ── Environment setup ─────────────────────────────────────────────────────────
    env = PandaLiberoSpatial(config=config, task_id=task_id)
    for run_id in range(1):
        frames = []

        rng = jax.random.PRNGKey(42 + run_id)
        print("=======================RESET=======================")
        state = env.reset(rng)

        # ── State machine ─────────────────────────────────────────────────────────────
        # States: approach_bowl -> descend_bowl -> grasp -> lift -> move_to_plate -> descend_plate -> release -> settle -> retreat
        PHASE_APPROACH_BOWL = 0
        PHASE_DESCEND_BOWL = 1
        PHASE_GRASP = 2
        PHASE_LIFT = 3
        PHASE_MOVE_TO_PLATE = 4
        PHASE_DESCEND_PLATE = 5
        PHASE_RELEASE = 6
        PHASE_SETTLE = 7
        PHASE_RETREAT = 8
        PHASE_DONE = 9

        phase_names = [
            "Approach bowl", "Descend to bowl", "Grasp", "Lift",
            "Move to plate", "Descend to plate", "Release", "Settle", "Retreat", "Done"
        ]

        phase = PHASE_APPROACH_BOWL
        counter = 0  # for timed phases (grasp, release, settle)

        # ── Renderer ──────────────────────────────────────────────────────────────────

        # --- Multi-camera renderer setup ---
        mj_model = env.mj_model
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Get all camera IDs and names
        cam_ids = [mj_model.camera(i).id for i in range(mj_model.ncam)]
        cam_names = [mj_model.camera(i).name for i in range(mj_model.ncam)]

        jit_step = jax.jit(env.step)
        # print("\nWarming up JIT...")
        _ = jit_step(state, jp.zeros(7))

        print("=======================INIT_OBJECTS=======================")
        for init_step_i in range(10):
            # print(f"Init step {init_step_i}")
            state = jit_step(state, jp.zeros(7))
            mj_data_render = mjx.get_data(mj_model, state.data)
            mujoco.mj_forward(mj_model, mj_data_render)
            images = []
            for cam_id in cam_ids:
                cam.fixedcamid = cam_id
                renderer.update_scene(mj_data_render, cam)
                img = renderer.render()
                images.append(img)
            # Concatenate all camera images horizontally
            side_by_side = np.concatenate(images, axis=1)
            frames.append(side_by_side)


        # ── Main loop ─────────────────────────────────────────────────────────────────
        MAX_STEPS = 500
        prev_phase = -1

        for step_i in range(MAX_STEPS):
            tip = get_tip(state)
            bowl_rf = get_bowl_rf(state, env)
            plate_rf = get_plate_rf(state, env)

            if phase != prev_phase:
                print(f"\nStep {step_i}: entering phase '{phase_names[phase]}'")
                print(f"  tip={tip}, bowl={bowl_rf}, plate={plate_rf}")
                prev_phase = phase
                counter = 0
                prev_error[:] = 0  # reset PID derivative on phase change
                error_integral[:] = 0  # reset PID integral on phase change

            # --- Decide action based on phase ---

            if phase == PHASE_APPROACH_BOWL:
                # Offset in Y by half diameter so fingers straddle the bowl
                grasp_y = bowl_rf[1] + BOWL_HALF_DIAMETER
                target_z = bowl_rf[2] + APPROACH_Z
                target = np.array([bowl_rf[0], grasp_y, target_z])
                action = pd_move(tip, target, GRIPPER_OPEN, dt=dt)
                if at_target(tip, target):
                    phase = PHASE_DESCEND_BOWL

            elif phase == PHASE_DESCEND_BOWL:
                # Descend straight down to grasp height relative to bowl
                grasp_y = bowl_rf[1] + BOWL_HALF_DIAMETER
                target_z = bowl_rf[2] + GRASP_Z
                target = np.array([bowl_rf[0], grasp_y, target_z])
                action = pd_move(tip, target, GRIPPER_OPEN, dt=dt)
                if at_target(tip, target):
                    phase = PHASE_GRASP

            elif phase == PHASE_GRASP:
                # Close gripper for fixed number of steps
                action = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSE])
                counter += 1
                if counter >= GRASP_STEPS:
                    phase = PHASE_LIFT

            elif phase == PHASE_LIFT:
                # Lift straight up to safe transport height (max of current safe lift or fixed base lift)
                target_z = max(LIFT_Z, bowl_rf[2] + 0.05)
                target = np.array([tip[0], tip[1], target_z])
                action = pd_move(tip, target, GRIPPER_CLOSE, dt=dt)
                if at_target(tip, target, z_tol=0.01):
                    phase = PHASE_MOVE_TO_PLATE

            elif phase == PHASE_MOVE_TO_PLATE:
                # Move horizontally to above the plate
                safe_z = max(LIFT_Z, plate_rf[2] + 0.15, bowl_rf[2] + 0.05)
                target = np.array([plate_rf[0], plate_rf[1], safe_z])
                action = pd_move(tip, target, GRIPPER_CLOSE, dt=dt)
                if at_target(tip, target):
                    phase = PHASE_DESCEND_PLATE

            elif phase == PHASE_DESCEND_PLATE:
                # Lower to release height above plate
                target_z = plate_rf[2] + RELEASE_Z
                target = np.array([plate_rf[0], plate_rf[1], target_z])
                action = pd_move(tip, target, GRIPPER_CLOSE, dt=dt)
                if at_target(tip, target, z_tol=0.01):
                    phase = PHASE_RELEASE

            elif phase == PHASE_RELEASE:
                # Open gripper
                action = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN])
                counter += 1
                if counter >= RELEASE_STEPS:
                    phase = PHASE_SETTLE

            elif phase == PHASE_SETTLE:
                # Wait for bowl to settle
                action = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN])
                counter += 1
                if counter >= SETTLE_STEPS:
                    phase = PHASE_RETREAT

            elif phase == PHASE_RETREAT:
                # Move up
                target = np.array([tip[0], tip[1], LIFT_Z + 0.1])
                action = pd_move(tip, target, GRIPPER_CLOSE, dt=dt)
                if at_target(tip, target, z_tol=0.01):
                    phase = PHASE_DONE

            else:
                break

            state = jit_step(state, action)


            if step_i % 1 == 0:
                mj_data_render = mjx.get_data(mj_model, state.data)
                mujoco.mj_forward(mj_model, mj_data_render)
                images = []
                for cam_id in cam_ids:
                    cam.fixedcamid = cam_id
                    renderer.update_scene(mj_data_render, cam)
                    img = renderer.render()
                    images.append(img)
                # Concatenate all camera images horizontally
                side_by_side = np.concatenate(images, axis=1)
                frames.append(side_by_side)

            # Log every 5 steps
            if step_i % 5 == 0:
                tip_now = get_tip(state)
                bowl_now = get_bowl_rf(state, env)
                ctrl7 = float(state.data.ctrl[7])  # gripper ctrl
                qpos_now = np.array(state.data.qpos)[env._robot_qposadr]
                print(f"  [{step_i:3d}] phase={phase_names[phase]:20s} tip=[{tip_now[0]:.4f},{tip_now[1]:.4f},{tip_now[2]:.4f}] bowl=[{bowl_now[0]:.4f},{bowl_now[1]:.4f},{bowl_now[2]:.4f}] grip_ctrl={ctrl7:.4f}")
                print(f"    qpos: {np.array2string(qpos_now, precision=3, separator=', ')}")

        # ── Final report ──────────────────────────────────────────────────────────────
        tip = get_tip(state)
        bowl_rf = get_bowl_rf(state, env)
        plate_rf = get_plate_rf(state, env)
        bowl_world = np.array(state.data.xpos[env._obj_body])
        plate_world = np.array(state.data.xpos[env._plate_body])
        dist = np.linalg.norm(bowl_world[:2] - plate_world[:2])

        print(f"\n{'='*60}")
        print(f"Completed at step {step_i}, phase: {phase_names[min(phase, PHASE_DONE)]}")
        print(f"Final tip (robot frame):  {tip}")
        print(f"Bowl (world):  {bowl_world}")
        print(f"Plate (world): {plate_world}")
        print(f"Bowl-plate XY distance: {dist:.4f}")
        print(f"Task {task_id} run {run_id}: {'SUCCESS' if dist < 0.05 else 'NEEDS TUNING'}: bowl {'is' if dist < 0.05 else 'is not'} on plate")

        renderer.close()

        out_path = f"renders/t_task{task_id}_runid{run_id}_new.mp4"
        os.makedirs("renders", exist_ok=True)
        mediapy.write_video(out_path, frames, fps=100)
        # print(f"\nSaved multi-camera video to {out_path} ({len(frames)} frames)")
