import mujoco
from mujoco import mjx

def render_frames(state_list, mj_model, height=360, width=480):
    # --- Multi-camera renderer setup ---
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # Get all camera IDs and names
    cam_ids = [mj_model.camera(i).id for i in range(mj_model.ncam)]

    frames_per_cam = {cam_id: [] for cam_id in cam_ids}
    for state in state_list:
        mj_data_render = mjx.get_data(mj_model, state.data)
        mujoco.mj_forward(mj_model, mj_data_render)
        for cam_id in cam_ids:
            cam.fixedcamid = cam_id
            renderer.update_scene(mj_data_render, cam)
            img = renderer.render()
            frames_per_cam[cam_id].append(img)
    return frames_per_cam
