"""Closed-loop pick-and-place: dynamically computes actions from live positions."""

import gc
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

from datetime import datetime

import jax
import jax.numpy as jp
import numpy as np
import mediapy

from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import (
    PandaLiberoSpatial,
    default_config,
)
from utils import render_frames

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

DT = 0.05
ROBOT_BASE = np.array([-0.6, 0.0, 0.912])
ACTION_SCALE = 0.005
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
GRIPPER_SEMI_CLOSE = 0
MAX_STEPS = 500

# Tolerances
XY_TOL = 0.01       # "close enough" in XY
Z_TOL = 0.008        # "close enough" in Z
GRASP_STEPS = 20     # steps to hold gripper closed
RELEASE_STEPS = 10   # steps to hold gripper open
SETTLE_STEPS = 10    # steps to let bowl settle after release

# Bowl geometry (from collision geoms: wall boxes at ~0.033-0.034 from center)
BOWL_HALF_DIAMETER = 0.034  # half diameter of the bowl

# Heights (robot base frame)
APPROACH_Z = 0.15    # hover height before descending
GRASP_Z = 0.03      # grasp height (proven to work in open-loop test)
LIFT_Z = 0.20        # lift height after grasping
RELEASE_Z = 0.07     # height above plate to release

# ── PID Controller ─────────────────────────────────────────────────────────────
# Action space: [-1, 1], each unit = ACTION_SCALE = 0.005m per step
# So to move 0.005m in one step, action = 1.0
# Kp: error in meters → action units. Kp=200 means 0.005m error → action=1.0
KP = 10  # proportional gain
KD = 0  # derivative gain (damps oscillation)
KI = 0    # integral gain (should be much smaller than KP)
INTEGRAL_WINDUP_LIMIT = 0.05  # max abs value for each integral component (meters)

prev_error = np.zeros(3)
error_integral = np.zeros(3)



def get_bowl_rf(state, env):
    """Bowl position in robot base frame."""
    return np.array(state.data.xpos[env._obj_body]) - ROBOT_BASE


def get_plate_rf(state, env):
    """Plate position in robot base frame."""
    return np.array(state.data.xpos[env._plate_body]) - ROBOT_BASE


def get_tip(state):
    """EE tip position in robot base frame."""
    return np.array(state.info["current_pos"])


def at_target(current, target, xy_tol=XY_TOL, z_tol=Z_TOL):
    """Check if tip is close enough to target."""
    return np.linalg.norm(current[:2] - target[:2]) < xy_tol and abs(current[2] - target[2]) < z_tol


def pid_move(current, target, gripper, dt=DT):
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

def run_task(task_id):
    config = default_config()
    # ── Environment setup ─────────────────────────────────────────────────────────
    env = PandaLiberoSpatial(config=config, config_overrides={"task_id": task_id})
    mj_model = env.mj_model
    jit_step = jax.jit(env.step)
    # Warm up JIT once per task
    _warmup_state = env.reset(jax.random.PRNGKey(0))
    _ = jit_step(_warmup_state, jp.zeros(7))
    del _warmup_state
    gc.collect()
    results = []
    for run_id in range(1):
        state_list = []

        rng = jax.random.PRNGKey(42 + run_id)
        state = env.reset(rng)
        phase = PHASE_APPROACH_BOWL
        counter = 0  # for timed phases (grasp, release, settle)

        
        # # ── Renderer ──────────────────────────────────────────────────────────────────

        # print("=======================INIT_OBJECTS=======================")
        for init_step_i in range(10):
            state = jit_step(state, jp.zeros(7))
        state_list.append(state)


        # ── Main loop ─────────────────────────────────────────────────────────────────
        prev_phase = -1

        total_reward = 0.0

        for step_i in range(MAX_STEPS):
            tip = get_tip(state)
            bowl_rf = get_bowl_rf(state, env)
            plate_rf = get_plate_rf(state, env)

            if phase != prev_phase:
                prev_phase = phase
                counter = 0
                prev_error[:] = 0  # reset PID derivative on phase change
                error_integral[:] = 0  # reset PID integral on phase change

            # --- Decide action based on phase ---

            if phase == PHASE_APPROACH_BOWL:
                # Offset in Y by half diameter so fingers straddle the bowl
                target_z = bowl_rf[2] + APPROACH_Z
                grasp_y = bowl_rf[1] + BOWL_HALF_DIAMETER
                target = np.array([bowl_rf[0], grasp_y, target_z])
                action = pid_move(tip, target, GRIPPER_OPEN, dt=DT)
                if at_target(tip, target):
                    phase = PHASE_DESCEND_BOWL

            elif phase == PHASE_DESCEND_BOWL:
                # Descend straight down to grasp height relative to bowl
                target_z = bowl_rf[2] + GRASP_Z
                grasp_y = bowl_rf[1] + BOWL_HALF_DIAMETER
                target = np.array([bowl_rf[0], grasp_y, target_z])
                action = pid_move(tip, target, GRIPPER_OPEN, dt=DT)
                if at_target(tip, target):
                    phase = PHASE_GRASP

            elif phase == PHASE_GRASP:
                # Close gripper for fixed number of steps
                action = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSE])
                counter += 1
                grasped_bowl_height = bowl_rf[2]
                if counter >= GRASP_STEPS:
                    phase = PHASE_LIFT

            elif phase == PHASE_LIFT:
                # Lift straight up to safe transport height
                target_z = max(LIFT_Z, grasped_bowl_height + 0.1)
                target = np.array([tip[0], tip[1], target_z])
                action = pid_move(tip, target, GRIPPER_CLOSE, dt=DT)
                if at_target(tip, target, z_tol=0.01):
                    phase = PHASE_MOVE_TO_PLATE

            elif phase == PHASE_MOVE_TO_PLATE:
                # Move horizontally to above the plate, staying at lift height
                tip_to_bowl = bowl_rf - tip
                safe_z = max(LIFT_Z, grasped_bowl_height + 0.1)
                target = np.array([plate_rf[0] - tip_to_bowl[0],
                                   plate_rf[1] - tip_to_bowl[1],
                                   safe_z])
                action = pid_move(tip, target, GRIPPER_CLOSE, dt=DT)
                if at_target(tip, target):
                    phase = PHASE_DESCEND_PLATE

            elif phase == PHASE_DESCEND_PLATE:
                # Lower to release height above plate
                tip_to_bowl = bowl_rf - tip
                target_z = plate_rf[2] + RELEASE_Z
                target = np.array([plate_rf[0] - tip_to_bowl[0],
                                   plate_rf[1] - tip_to_bowl[1],
                                   target_z - tip_to_bowl[2]])
                action = pid_move(tip, target, GRIPPER_CLOSE, dt=DT)
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
                action = pid_move(tip, target, GRIPPER_SEMI_CLOSE, dt=DT)
                if at_target(tip, target, z_tol=0.01):
                    phase = PHASE_DONE

            else:
                break

            state = jit_step(state, action)
            total_reward += state.reward
            state_list.append(state)
            if state.info["_steps"] == 0:
                print(f"Episode ended at step {step_i}")
                break


        # ── Final report ──────────────────────────────────────────────────────────────
        tip = get_tip(state)
        bowl_rf = get_bowl_rf(state, env)
        plate_rf = get_plate_rf(state, env)
        bowl_world = np.array(state.data.xpos[env._obj_body])
        plate_world = np.array(state.data.xpos[env._plate_body])
        dist = np.linalg.norm(bowl_world - plate_world)

        success = dist < 0.05
        # if not success:
        # frames_per_cam = render_frames(state_list, mj_model)
        print(f"Task {task_id} run {run_id}: {'SUCCESS' if success else 'NEEDS TUNING'}: bowl {'is' if success else 'is not'} on plate")
        results.append((task_id, run_id, success, float(dist), float(total_reward)))
        # mediapy.write_video(f"video_task_id{task_id}_run{run_id}.mp4", list(frames_per_cam.values())[0], fps=20)

        # Save state_list 
        import pickle
        path = "./"
        with open(f"{path}/states_tid{task_id}_rid{run_id}.pkl", "wb") as f:
            pickle.dump(state_list, f)


        for s in state_list:
            del s
        del state_list
        gc.collect()

    return results


if __name__ == "__main__":

    all_results = []
    for task_id in range(1):
        task_results = run_task(task_id)
        all_results.extend(task_results)

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for task_id, run_id, success, dist, total_reward in all_results:
        print(f"Task {task_id} run {run_id}: {'SUCCESS' if success else 'FAIL'} (dist={dist:.4f}, total_reward={total_reward:.4f})")
