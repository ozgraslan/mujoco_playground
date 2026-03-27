"""Render the LIBERO spatial scene using PandaLiberoSpatial environment."""

import os
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from pathlib import Path
from PIL import Image

from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import (
    PandaLiberoSpatial,
    _SPATIAL_TASKS,
)

out_dir = Path("renders")
out_dir.mkdir(exist_ok=True)

rng = jax.random.PRNGKey(42)

for task_id, (desc, _, _) in _SPATIAL_TASKS.items():
    env = PandaLiberoSpatial(task_id=task_id)
    renderer = mujoco.Renderer(env._mj_model, height=480, width=640)

    rng, rng_reset = jax.random.split(rng)
    state = env.reset(rng_reset)

    # Convert mjx.Data back to mj.Data for settling and rendering
    mj_data = mjx.get_data(env._mj_model, state.data)
    mujoco.mj_forward(env._mj_model, mj_data)

    # Settle objects on the table using plain MuJoCo
    for _ in range(500):
        mujoco.mj_step(env._mj_model, mj_data)

    for cam_name in ["agentview", "sideview"]:
        renderer.update_scene(mj_data, camera=cam_name)
        img = renderer.render()
        out_path = out_dir / f"render_task{task_id}_{cam_name}.png"
        Image.fromarray(img).save(out_path)

    bowl_pos = mj_data.qpos[env._obj_qposadr : env._obj_qposadr + 3]
    print(f"Task {task_id}: {desc}")
    print(f"  Bowl at ({bowl_pos[0]:.3f}, {bowl_pos[1]:.3f}, {bowl_pos[2]:.3f}) -> {out_path}")
    renderer.close()

print("Done!")
