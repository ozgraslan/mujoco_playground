"""In-process grid search over CEM hyperparameters.

Runs all configs in a single process to avoid GPU memory issues from
subprocess spawning. The env is created once; the CEM planner is
re-built per config.
"""

import os
os.environ["MUJOCO_GL"] = "egl"
import gc

import itertools
import time


import numpy as np
import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx
import warp as wp

import wandb

# Cap how much VRAM a single config can use (samples * horizon * substeps)
# 512 * 20 * 10 substeps is ~100k parallel steps — too much for most GPUs.
# Rough budget: num_samples * horizon. Tune based on your GPU.
MAX_SAMPLES_X_HORIZON = 5120  # e.g. 256*20=5120 OK, 256*30=7680 skip

from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import (
    PandaLiberoSpatial,
)
from cem_planning import CEMConfig, make_cem_planner

# ---- Define parameter grid ----
param_grid = {
    "num_samples": [64, 128, 256],
    "horizon": [5, 10, 20],
    "num_iterations": [3, 5, 8, 10],
    "elite_frac": [0.05, 0.1, 0.2],
    "init_std": [0.1, 0.3, 0.5],
    "momentum": [0.0, 0.1, 0.3],
}
## add action repeating
## before and after time 

# Fixed settings
TASK_ID = 0
SEED = 42
NUM_PLANNING_STEPS = 500
VIDEO_DIR = "sweep_videos"
OUTPUT_CSV = "cem_sweep_results.csv"
SAVE_VIDEOS = True  # set True to save a video per config (slow)
CAMERA = "agentview"

# ---- Init wandb ----
wandb.init(
    project="cem-libero-sweep",
    config={
        "param_grid": param_grid,
        "task_id": TASK_ID,
        "seed": SEED,
        "num_planning_steps": NUM_PLANNING_STEPS,
        "max_samples_x_horizon": MAX_SAMPLES_X_HORIZON,
    },
)

# ---- Create env once ----
print("Creating environment...")
env = PandaLiberoSpatial(
    task_id=TASK_ID,
    config_overrides={"naconmax": 16384, "naccdmax": 16384, "njmax": 1024},
) # "impl": "jax", 
step_fn = jax.jit(env.step)
reset_fn = jax.jit(env.reset)
print(f"Action dim: {env.action_size}, Obs dim: {env.observation_size}")


def run_one_config(cem_cfg: CEMConfig, env, seed: int):
    """Run CEM-MPC for one config, return (total_reward, total_time, jit_time, steps)."""
    plan_fn = make_cem_planner(env, cem_cfg)

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    # Settle objects
    zero_action = jnp.zeros(env.action_size)
    for _ in range(10):
        state = step_fn(state, zero_action)

    # JIT warmup
    rng, plan_rng = jax.random.split(rng)
    t_jit = time.time()
    action, mean, std, info = plan_fn(state, plan_rng, None, None)
    jax.block_until_ready(action)
    jit_time = time.time() - t_jit

    # MPC loop
    total_reward = 0.0
    prev_mean, prev_std = mean, std
    trajectory_states = [state] if SAVE_VIDEOS else None

    t0 = time.time()
    for i in range(NUM_PLANNING_STEPS):
        rng, plan_rng = jax.random.split(rng)
        action, prev_mean, prev_std, info = plan_fn(
            state, plan_rng, prev_mean, prev_std
        )
        state = step_fn(state, action)
        if SAVE_VIDEOS:
            trajectory_states.append(state)
        r = float(state.reward)
        total_reward += r
        if float(state.done):
            break

    total_time = time.time() - t0
    steps_done = i + 1
    return total_reward, total_time, jit_time, steps_done, trajectory_states


def render_to_wandb_video(env, trajectory_states):
    """Render trajectory and return wandb.Video from numpy array."""
    renderer = mujoco.Renderer(env._mj_model, height=480, width=640)
    frames = []
    for s in trajectory_states:
        mj_data = mjx.get_data(env._mj_model, s.data)
        mujoco.mj_forward(env._mj_model, mj_data)
        renderer.update_scene(mj_data, camera=CAMERA)
        frames.append(renderer.render().copy())
    renderer.close()
    # wandb.Video expects (T, C, H, W) for numpy input
    video_array = np.stack(frames).transpose(0, 3, 1, 2)
    return wandb.Video(video_array, fps=20)


# ---- Run sweep ----
keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
print(f"\nTotal configurations: {len(combos)}\n")

results = []
for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    cem_cfg = CEMConfig(**params)
    tag = "_".join(f"{k}{v}" for k, v in params.items())

    budget = params["num_samples"] * params["horizon"]
    if budget > MAX_SAMPLES_X_HORIZON:
        print(f"[{idx+1}/{len(combos)}] {params} ... SKIPPED (budget {budget} > {MAX_SAMPLES_X_HORIZON})")
        row = {**params, "total_reward": None, "total_time": None,
               "jit_time": None, "steps": None}
        results.append(row)
        continue

    print(f"[{idx+1}/{len(combos)}] {params}", end=" ... ", flush=True)
    try:
        reward, plan_time, jit_time, steps, traj = run_one_config(cem_cfg, env, SEED)
        print(f"reward={reward:.4f}, time={plan_time:.1f}s, jit={jit_time:.1f}s, steps={steps}")

        row = {**params, "total_reward": reward, "total_time": plan_time,
               "jit_time": jit_time, "steps": steps}

        # Log to wandb
        wandb_log = {**params, "total_reward": reward, "total_time": plan_time,
                     "jit_time": jit_time, "steps": steps, "run_idx": idx}
        if SAVE_VIDEOS and traj:
            wandb_log["video"] = render_to_wandb_video(env, traj)
        wandb.log(wandb_log)
    except Exception as e:
        print(f"FAILED: {e}")
        row = {**params, "total_reward": None, "total_time": None,
               "jit_time": None, "steps": None}

    # Clear JIT caches and warp GPU memory for next config
    jax.clear_caches()
    wp.synchronize_device("cuda:0")
    gc.collect()

    results.append(row)

# ---- Save results to wandb ----
fieldnames = keys + ["total_reward", "total_time", "jit_time", "steps"]

# Log as wandb Table for interactive exploration
table = wandb.Table(columns=fieldnames)
for r in results:
    table.add_data(*[r.get(k) for k in fieldnames])
wandb.log({"results_table": table})

print("\nResults uploaded to wandb")

# ---- Print top 10 ----
valid = [r for r in results if r["total_reward"] is not None]
valid.sort(key=lambda r: r["total_reward"], reverse=True)
print("\nTop 10 configurations by reward:")
for i, r in enumerate(valid[:10]):
    params_str = ", ".join(f"{k}={r[k]}" for k in keys)
    print(f"  {i+1}. reward={r['total_reward']:.4f} time={r['total_time']:.1f}s | {params_str}")

wandb.finish()
