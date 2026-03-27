import os
os.environ["MUJOCO_GL"] = "egl"
import jax
import mujoco
from mujoco import mjx
import numpy as np
import mediapy as media
from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import PandaLiberoSpatial, default_config

# Directory to save images
SAVE_DIR = "renders/reset_agentview_obs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Number of seeds
N = 1

# Camera name for agentview
CAMERA_NAME = "agentview"

# Initialize environment
config = default_config()
for task_id in range(9):
    env = PandaLiberoSpatial(config=config, task_id=task_id)
    mj_model = env.mj_model

    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mj_model.camera("agentview").id

    for i in range(N):
        # Set random seed
        rng = jax.random.PRNGKey(i)
        state = env.reset(rng)
        mj_data = mjx.get_data(mj_model, state.data)
        frames = []
        for step_id in range(100):
            mujoco.mj_step(mj_model, mj_data)
    
            renderer.update_scene(mj_data, cam)
            img = renderer.render()
            frames.append(np.asarray(img))

        # Save image
        out_path = os.path.join(SAVE_DIR, f"reset_obs_task_{task_id}_seed{i}.mp4")
        # media.write_image(out_path, np.asarray(img))
        media.write_video(out_path, frames, fps=20)
        print(f"Saved: {out_path}")

        # Print plate position
        plate_body_id = mj_model.body('plate_1').id
        plate_pos = mj_data.xpos[plate_body_id]
        print(f" Task {task_id} Seed {i}: Plate position: {plate_pos}")
    renderer.close()