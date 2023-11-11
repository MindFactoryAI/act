from data_utils import save_episode, load_hdf5
from uuid import uuid4
from aloha_scripts.constants import CAM_NAMES
import numpy as np
from dm_env._environment import TimeStep
from dm_env import StepType
from utils import CompressedEpisodicDataset, get_norm_stats
import cv2
import torch


def make_timestep():
    """
        Action
        space: [left_arm_qpos(6),  # absolute joint position
                left_gripper_positions(1),  # normalized gripper position (0: close, 1: open)
                right_arm_qpos(6),  # absolute joint position
                right_gripper_positions(1), ]  # normalized gripper position (0: close, 1: open)

        Observation
        space: {"qpos": Concat[left_arm_qpos(6),  # absolute joint position
                        left_gripper_position(1),  # normalized gripper position (0: close, 1: open)
                        right_arm_qpos(6),  # absolute joint position
                        right_gripper_qpos(1)]  # normalized gripper position (0: close, 1: open)

                "qvel": Concat[left_arm_qvel(6),  # absolute joint velocity (rad)
                        left_gripper_velocity(1),  # normalized gripper velocity (pos: opening, neg: closing)
                        right_arm_qvel(6),  # absolute joint velocity (rad)
                        right_gripper_qvel(1)]  # normalized gripper velocity (pos: opening, neg: closing)
                "images": {"cam_high": (480x640x3),  # h, w, c, dtype='uint8'
                           "cam_low": (480x640x3),  # h, w, c, dtype='uint8'
                           "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                           "cam_right_wrist": (480x640x3)}  # h, w, c, dtype='uint8'
    """

    raw_image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    retval, compressed_image = cv2.imencode(".jpg", raw_image)

    obs = {
        "qpos": np.random.randn(14, ),
        "qvel": np.random.randn(14, ),
        "effort": np.random.randn(14, ),
        "images": {
            "cam_high": raw_image,  # h, w, c, dtype='uint8'#
            "cam_low": raw_image,  # h, w, c, dtype='uint8'
            "cam_left_wrist": raw_image,  # h, w, c, dtype='uint8'
            "cam_right_wrist": raw_image  # h, w, c, dtype='uint8'
        },
        "images_compressed": {
            "cam_high": compressed_image,
            "cam_low": compressed_image,
            "cam_left_wrist": compressed_image,
            "cam_right_wrist": compressed_image
        }
    }

    return TimeStep(StepType.MID, 0., None, obs)


def make_action():
    return np.random.randn(14)


def make_trajectory(length, rating):
    return [make_timestep()] * length, [make_action()] * length, make_timestep(), str(uuid4()), rating


def test_save_and_dataload():
    prediction_len = 100
    dataset_path = 'episode_0'
    max_timesteps = 5

    timesteps, actions, terminal_state, policy_guid, rating = make_trajectory(4, 5)

    save_episode(dataset_path, policy_guid, CAM_NAMES, max_timesteps, timesteps, actions, terminal_state,
                 result=None, rating=rating, policy_info=None, policy_index=None)

    norm_stats = get_norm_stats('.', 1)

    ds = CompressedEpisodicDataset(prediction_len, [0], '.', CAM_NAMES, norm_stats, sample_full_episode=True)

    image_data, qpos_data, action_data, is_pad, monte_carlo_estimate = ds[0]

    assert len(action_data) == prediction_len
    assert image_data.shape == (4, 3, 480, 640)
    assert qpos_data.shape == (14, )
    is_pad_gt = torch.ones_like(is_pad)
    is_pad_gt[0:4] = 0
    assert (~is_pad).sum() == 4
    assert (~torch.logical_xor(is_pad, is_pad_gt)).all()


def test_save_and_load():
    dataset_path = 'episode_0'
    max_timesteps = 5

    timesteps, actions, terminal_state, policy_guid, rating = make_trajectory(4, 5)

    save_episode(dataset_path, policy_guid, CAM_NAMES, max_timesteps, timesteps, actions, terminal_state,
                 result=None, rating=rating, policy_info=None, policy_index=None)

    qpos, qvel, effort, action, image_dict = load_hdf5('.', 'episode_0')

    assert qpos.shape == (5, 14)
    assert len(image_dict['cam_high']) == 5
    assert action.shape == (4, 14)
    assert image_dict['cam_high'][0].shape == (480, 640, 3)