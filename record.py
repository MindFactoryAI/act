import argparse
from aloha_scripts.constants import DT, TASK_CONFIGS, get_start_arm_pose
from aloha_scripts.record_episodes import get_auto_index, capture_one_episode, opening_ceremony
from aloha_scripts.robot_utils import move_arms
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from real_env import make_real_env, get_action


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']
    current_limit = args['current_limit']

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False, task=args['task_name'])

    reboot = True  # always reboot on the first episode

    while True:

        if args['episode_idx'] is not None:
            episode_idx = args['episode_idx']
        else:
            episode_idx = get_auto_index(dataset_dir)
        overwrite = True
        dataset_name = f'episode_{episode_idx}'
        print(dataset_name + '\n')
        start_left_arm_pose, start_right_arm_pose = get_start_arm_pose(args['task_name'])
        opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, reboot,
                         current_limit, start_left_arm_pose, start_right_arm_pose)
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
                                         master_bot_left, master_bot_right, env, reboot, current_limit, start_left_arm_pose, start_right_arm_pose)
        if is_healthy and args['episode_idx'] is not None:
            break
        reboot = args['reboot_every_episode']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--reboot_every_episode', action='store_true', help='Episode index.', default=False, required=False)
    parser.add_argument('--current_limit', help='Episode index.', default=500, required=False)
    main(vars(parser.parse_args()))
    # debug()
