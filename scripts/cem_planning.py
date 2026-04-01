"""CEM (Cross-Entropy Method) planning with vectorized LIBERO spatial environment."""

import os
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
import time
import tyro
import mujoco
from mujoco import mjx
from dataclasses import dataclass, field

from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import (
    PandaLiberoSpatial,
)


@dataclass
class CEMConfig:
    """CEM hyperparameters."""
    horizon: int = 20             # planning horizon (number of steps)
    elite_frac: float = 0.1       # fraction of top samples to keep
    num_samples: int = 256        # number of action sequences to sample per iteration
    init_std: float = 0.3         # initial std for action sampling
    num_iterations: int = 5       # CEM iterations per planning step
    min_std: float = 0.01         # minimum std (prevents collapse)
    momentum: float = 0.1         # momentum for mean/std updates (0 = no momentum)
    action_low: float = -1.0      # action clipping lower bound
    action_high: float = 1.0      # action clipping upper bound
    action_repeat: int = 1        # apply each planned action this many env steps


@dataclass
class RunConfig:
    """Top-level configuration."""
    cem: CEMConfig = field(default_factory=CEMConfig)
    task_id: int = 0              # LIBERO spatial task variant (0-8)
    num_planning_steps: int = 500 # number of MPC steps to run
    seed: int = 42                # random seed
    video_path: str = "cem_trajectory.mp4"  # output video path
    camera: str = "agentview"     # camera name for rendering
    video_height: int = 480       # video frame height
    video_width: int = 640        # video frame width
    video_fps: int = 20           # video frames per second


def make_cem_planner(env: PandaLiberoSpatial, cfg: CEMConfig):
    """Creates JIT-compiled CEM planning functions.

    Returns:
        plan_fn: (state, rng, prev_mean, prev_std) -> (best_action, mean, std, info)
    """
    action_dim = env.action_size
    num_elites = max(1, int(cfg.num_samples * cfg.elite_frac))

    def rollout_actions(state, action_sequence):
        """Roll out a single action sequence from a given state.
        
        Args:
            state: environment State (single, not batched)
            action_sequence: (horizon, action_dim) array
        
        Returns:
            total_reward: scalar cumulative reward
        """
        def repeat_action(carry, _):
            """Apply the same action once (used for action_repeat)."""
            st, total_r, alive, action = carry
            next_st = env.step(st, action)
            total_r = total_r + next_st.reward * alive
            alive = alive * (1.0 - next_st.done)
            return (next_st, total_r, alive, action), None

        def scan_fn(carry, action):
            st, total_r, alive = carry
            (st, total_r, alive, _), _ = jax.lax.scan(
                repeat_action, (st, total_r, alive, action), None, length=cfg.action_repeat
            )
            return (st, total_r, alive), None

        (final_state, total_reward, _), _ = jax.lax.scan(
            scan_fn, (state, 0.0, 1.0), action_sequence
        )
        return total_reward

    # vmap over num_samples action sequences, same state
    batched_rollout = jax.vmap(rollout_actions, in_axes=(None, 0))

    def cem_iteration(carry, _):
        """Single CEM iteration: sample, evaluate, refit."""
        mean, std, state, rng = carry
        rng, sample_rng = jax.random.split(rng)

        # Sample action sequences: (num_samples, horizon, action_dim)
        noise = jax.random.normal(sample_rng, (cfg.num_samples, cfg.horizon, action_dim))
        actions = mean[None, :, :] + std[None, :, :] * noise
        actions = jnp.clip(actions, cfg.action_low, cfg.action_high)

        # Evaluate all sequences in parallel
        rewards = batched_rollout(state, actions)  # (num_samples,)

        # Select elites
        elite_indices = jnp.argsort(rewards)[-num_elites:]
        elites = actions[elite_indices]  # (num_elites, horizon, action_dim)

        # Refit distribution
        new_mean = jnp.mean(elites, axis=0)
        new_std = jnp.maximum(jnp.std(elites, axis=0), cfg.min_std)

        # Apply momentum
        mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
        std = cfg.momentum * std + (1 - cfg.momentum) * new_std

        best_reward = rewards[elite_indices[-1]]
        return (mean, std, state, rng), best_reward

    def plan(state, rng, prev_mean=None, prev_std=None):
        """Run CEM planning from current state.

        Args:
            state: current env State (single, unbatched)
            rng: JAX PRNGKey
            prev_mean: optional warm-start mean (horizon, action_dim)
            prev_std: optional warm-start std (horizon, action_dim)

        Returns:
            best_action: (action_dim,) action to execute
            shifted_mean: (horizon, action_dim) warm-start for next call
            shifted_std: (horizon, action_dim) warm-start for next call
            info: dict with best_rewards per iteration
        """
        # Initialize or warm-start
        if prev_mean is None:
            mean = jnp.zeros((cfg.horizon, action_dim))
        else:
            # Shift by 1 step: drop first, append zeros
            mean = jnp.concatenate([prev_mean[1:], jnp.zeros((1, action_dim))], axis=0)

        if prev_std is None:
            std = jnp.full((cfg.horizon, action_dim), cfg.init_std)
        else:
            std = jnp.concatenate(
                [prev_std[1:], jnp.full((1, action_dim), cfg.init_std)], axis=0
            )

        init_carry = (mean, std, state, rng)
        (final_mean, final_std, _, _), best_rewards = jax.lax.scan(
            cem_iteration, init_carry, None, length=cfg.num_iterations
        )

        best_action = final_mean[0]
        best_action = jnp.clip(best_action, cfg.action_low, cfg.action_high)

        return best_action, final_mean, final_std, {"best_rewards": best_rewards}

    return jax.jit(plan)


def main(cfg: RunConfig):
    cem_cfg = cfg.cem
    print(f"CEM Config: {cem_cfg}")
    print(f"Task ID: {cfg.task_id}, Planning steps: {cfg.num_planning_steps}")

    # ---- Create env ----
    env = PandaLiberoSpatial(
        task_id=cfg.task_id,
        config_overrides={"impl": "warp", "naconmax": 16384 * 2, "naccdmax": 16384 * 2, "njmax": 1024*2}
    )
    print(f"Action dim: {env.action_size}, Obs dim: {env.observation_size}")

    # ---- Build planner ----
    plan_fn = make_cem_planner(env, cem_cfg)

    # ---- Reset ----
    rng = jax.random.PRNGKey(cfg.seed)
    rng, reset_rng = jax.random.split(rng)
    reset_fn = jax.jit(env.reset)
    state = reset_fn(reset_rng)

    # Settle objects on the table with zero actions
    step_fn = jax.jit(env.step)
    zero_action = jnp.zeros(env.action_size)
    for _ in range(10):
        state = step_fn(state, zero_action)

    # ---- Warmup JIT ----
    print("JIT compiling CEM planner (first call)...")
    rng, plan_rng = jax.random.split(rng)
    t0 = time.time()
    action, mean, std, info = plan_fn(state, plan_rng, None, None)
    jax.block_until_ready(action)
    print(f"JIT compile + first plan: {time.time() - t0:.2f}s")

    # ---- Run MPC loop ----
    total_reward = 0.0
    prev_mean, prev_std = mean, std
    trajectory_states = [state]  # collect states for rendering

    print("\nRunning CEM-MPC...")
    t0 = time.time()
    for i in range(cfg.num_planning_steps):
        rng, plan_rng = jax.random.split(rng)
        action, prev_mean, prev_std, info = plan_fn(
            state, plan_rng, prev_mean, prev_std
        )
        # Apply the planned action for action_repeat steps
        for _ in range(cem_cfg.action_repeat):
            state = step_fn(state, action)
            trajectory_states.append(state)
        r = float(state.reward)
        total_reward += r

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            best_r = float(info["best_rewards"][-1])
            print(
                f"  Step {i+1:3d} | reward={r:.4f} | cumulative={total_reward:.4f} "
                f"| CEM best={best_r:.4f} | {elapsed/(i+1):.3f}s/step"
            )

        if float(state.done):
            print(f"  Episode done at step {i+1}")
            break

    elapsed = time.time() - t0
    print(f"\nDone. Total reward: {total_reward:.4f}")
    print(f"Avg time per step: {elapsed / min(i+1, cfg.num_planning_steps):.3f}s")
    print(f"Total planning time: {elapsed:.2f}s")

    # ---- Render trajectory to video ----
    print(f"\nRendering {len(trajectory_states)} frames to {cfg.video_path}...")
    import cv2
    renderer = mujoco.Renderer(env._mj_model, height=cfg.video_height, width=cfg.video_width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(cfg.video_path, fourcc, cfg.video_fps, (cfg.video_width, cfg.video_height))
    for s in trajectory_states:
        mj_data = mjx.get_data(env._mj_model, s.data)
        mujoco.mj_forward(env._mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cfg.camera)
        frame = renderer.render()
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    renderer.close()
    print(f"Saved video: {cfg.video_path} ({len(trajectory_states)} frames, {cfg.video_fps} fps)")


if __name__ == "__main__":
    main(tyro.cli(RunConfig))
