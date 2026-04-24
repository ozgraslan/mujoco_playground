import os
import math
import pickle 
import json

import numpy as np
from jax import numpy as jnp

import torch
import mediapy

from mujoco_playground._src.manipulation.franka_emika_panda.libero_spatial import (
    PandaLiberoSpatial,

    default_config,
)
from utils import render_frames


PI = np.pi
EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
def vec(values):
    """
    Converts value tuple into a numpy vector.

    Args:
        values (n-array): a tuple of numbers

    Returns:
        np.array: vector of given values
    """
    return np.array(values, dtype=np.float32)

def mat2euler(rmat, axes="sxyz"):
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.asarray(rmat, dtype=np.float32)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az))



if __name__ == "__main__":
    rgb_skip = 4  # sample every 4th frame for videos to reduce length
    task_id = 0
    run_id = 0
    size = (192,320)
    threshold = 1e-4
    path = "./"
    state_list = pickle.load(open(f"{path}/states_tid{task_id}_rid{run_id}.pkl", "rb"))

    config = default_config()
    env = PandaLiberoSpatial(config=config, config_overrides={"task_id": task_id})
    mj_model = env.mj_model


    frames_per_cam = render_frames(state_list, mj_model, height=size[0], width=size[1])
    sampled_videos = []
    for index, frames in enumerate(frames_per_cam.values()):
        frames = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0*2-1
        frames = frames[::rgb_skip]  # sample every 4th frame to reduce video length
        x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
        resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
        resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        mediapy.write_video(f"video_{index}.mp4", resize_video, fps=5)



    robot_pose = []
    robot_joints = []
    for state in state_list:
        joint_pos = state.info['joint_position'][:7]
        gripper_pos =  1 - (state.info['joint_position'][-1] * 25)
        joint_all = np.concatenate([joint_pos, [gripper_pos]])
        absolute_pose = np.concatenate([np.array(state.info['current_pos']), mat2euler(np.array(state.info['current_rot'])), [gripper_pos]])
        joint_all = np.where(np.abs(joint_all) < threshold, 0.0, joint_all)
        absolute_pose = np.where(np.abs(absolute_pose) < threshold, 0.0, absolute_pose)
        robot_joints.append(joint_all.tolist())
        robot_pose.append(absolute_pose.tolist())


    print("Before sampling sizes:", len(robot_pose), len(robot_joints))
    sampled_robot_pose = robot_pose[::rgb_skip]
    sampled_robot_joints = robot_joints[::rgb_skip]
    print("After sampling sizes:", len(sampled_robot_pose), len(sampled_robot_joints))

    bowl_world = np.array(state_list[-1].data.xpos[env._obj_body])
    plate_world = np.array(state_list[-1].data.xpos[env._plate_body])
    dist = np.linalg.norm(bowl_world - plate_world)
    success = dist < 0.05

    texts = [env.task_description]

    info = {
        "texts": texts,
        "episode_id": task_id * 2 + run_id,
        "success": int(success),
        "video_length": frames.shape[0],
        "state_length": len(sampled_robot_pose),
        "raw_length": len(robot_pose),
        "states": sampled_robot_pose,
        "joints": sampled_robot_joints,
        "videos": [
            {"video_path": "video_0.mp4"},
            {"video_path": "video_1.mp4"},
            {"video_path": "video_2.mp4"}
        ],
    }
    with open(f"annotation_{task_id}_{run_id}.json", "w") as f:
        json.dump(info, f, indent=2)
