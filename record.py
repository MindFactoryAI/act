import argparse
from aloha_scripts.constants import DT, TASK_CONFIGS, get_start_arm_pose
from aloha_scripts.record_episodes import get_auto_index, capture_one_episode, opening_ceremony, wait_for_start
from aloha_scripts.robot_utils import move_arms
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from real_env import make_real_env, get_action
from imitate_episodes import execute_policy_on_env, load_policy_and_stats


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

    state_dim = 14
    policy_config = {'lr': 1e-5,
                     'num_queries': args['chunk_size'],
                     'kl_weight': 0.0,
                     'hidden_dim': args['hidden_dim'],
                     'dim_feedforward': args['dim_feedforward'],
                     'lr_backbone': 1e-5,
                     'backbone': 'resnet18',
                     'enc_layers': 4,
                     'dec_layers': 7,
                     'nheads': 8,
                     'camera_names': camera_names,
                     }

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

        initial_state = env.reset()

        # execute first policy
        task = TASK_CONFIGS['grasp_battery']
        policy, stats = load_policy_and_stats(policy_config, '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt')
        wait_for_start(master_bot_left, master_bot_right, human_takeover=False)
        qpos_history, image_list, qpos_list, target_qpos_list, rewards, last_state = \
            execute_policy_on_env(policy, env, initial_state, task['episode_len'], state_dim, stats, camera_names,
                                  master_bot_left=master_bot_left, master_bot_right=master_bot_right)
        wait_for_start(master_bot_left, master_bot_right)
        is_healthy = capture_one_episode(last_state, DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite,
                                         master_bot_left, master_bot_right, env)
        if is_healthy and args['episode_idx'] is not None:
            break
        reboot = args['reboot_every_episode']


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--reboot_every_episode', action='store_true', help='Episode index.', default=False, required=False)
    parser.add_argument('--current_limit', type=int, help='gripper current limit', default=300, required=False)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--dim_feedforward', type=int, default=3200)

    # dummy argument to make DETR happy
    parser.add_argument('--policy_class', type=str, default='ACT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default='dummy')

    main(vars(parser.parse_args()))
    # debug()
