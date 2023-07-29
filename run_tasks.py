import argparse
from pathlib import Path

from aloha_scripts.constants import START_ARM_POSE, TASK_CONFIGS
from aloha_scripts.record_episodes import opening_ceremony
from data_utils import save_episode, validate_dataset, get_auto_index, Episode
from robot_utils import wait_for_input, LEFT_HANDLE_CLOSED, RIGHT_HANDLE_CLOSED, LEFT_HANDLE_OPEN, RIGHT_HANDLE_OPEN, BOTH_OPEN
from checkpoint import CheckPointInfo
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from real_env import make_real_env
from primitives import ACTPrimitive, CapturePrimitive
from matplotlib import pyplot as plt
import numpy as np

ROUTINES = {
    'grasp_battery': {
        'sequence': [
            ACTPrimitive('grasp_battery', '/mnt/magneto/checkpoints/grasp_battery/worldly-puddle-13/policy_min_val_loss0.27740.ckpt'),
        ]
    },
    'drop_battery_in_slot': {
        'sequence': [
            ACTPrimitive('grasp_battery'),
            ACTPrimitive('drop_battery_in_slot_only',
                         '/mnt/magneto/checkpoints/drop_battery_in_slot_only/silvery-cherry-27/policy_min_val_loss0.23052.ckpt',
                      )
        ]
    },

    'drop_battery_in_slot_dev': {
        'sequence': [
            ACTPrimitive('grasp_battery',
                         '/mnt/magneto/checkpoints/grasp_battery/divine-sun-8/policy_min_val_loss0.28309.ckpt'),
            ACTPrimitive('drop_battery_in_slot_only',
                         '/mnt/magneto/checkpoints/drop_battery_in_slot_only/denim-smoke-20/policy_min_val_loss0.32010.ckpt'
)
        ]
    },

    'push_battery_in_slot_only': {
        'sequence': [
            ACTPrimitive('push_battery_in_slot',
                         '/mnt/magneto/checkpoints/push_battery_in_slot/peach-disco-3/policy_best_inv_learning_error_0.05223.ckpt'
                      # '/mnt/magneto/checkpoints/push_battery_in_slot/peach-disco-3/policy_min_val_loss0.33072.ckpt'
                      # '/mnt/magneto/checkpoints/push_battery_in_slot/dummy-osa5na9u/policy_min_val_loss0.31343.ckpt'
                      )
        ]
    },
    'slot_battery': {
        'sequence': [
            ACTPrimitive('grasp_battery',
                         '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt'),
            ACTPrimitive('drop_battery_in_slot_only',
                         '/mnt/magneto/checkpoints/drop_battery_in_slot_only/noble-shape-2/policy_best_inv_learning_error_0.04085.ckpt'),
            ACTPrimitive('push_battery_in_slot',
                         '/mnt/magneto/checkpoints/push_battery_in_slot/dummy-osa5na9u/policy_min_val_loss0.31343.ckpt')
        ]
    },
}


def main(args):
    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    reboot = True  # always reboot on the first episode

    sequence = ROUTINES[args.routine_name]['sequence']
    human_capture_sequence = [CapturePrimitive(prim.task_name) for prim in sequence]

    if args.capture_mode == "ALL":
        capture_flags = [True] * len(sequence)
    elif args.capture_mode == "LAST":
        capture_flags = [False] * (len(sequence) - 1) + [True]
    elif args.capture_mode is None:
        capture_flags = [False] * len(sequence)
    else:
        raise Exception('valid capture modes are ALL or LAST')

    plt.ion()
    fig, ax = plt.subplots(1, len(sequence), figsize=(12 * len(sequence), 12), dpi=80)
    ax_initial_state_img = []
    # ax = ax if isinstance(ax, type(list)) else [ax]
    for i in range(len(sequence)):
        ax_initial_state_img.append(ax[i].imshow(np.zeros((640, 480))))
    fig.tight_layout()
    fig.canvas.draw()
    plt.draw()
    plt.pause(1)

    def update_panel(sequence_i, episode_path):
        frame = Episode(f'{episode_path}.hdf5').get_frame(0, "RGB")
        cam_low = Episode(f'{episode_path}.hdf5').split_frame(frame)['cam_low']
        ax_initial_state_img[sequence_i].set_data(np.fliplr(cam_low.swapaxes(0, 1)))
        plt.pause(1)

    while True:

        opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, reboot,
                         args.current_limit, START_ARM_POSE[:6], START_ARM_POSE[:6])

        initial_state = env.reset()

        handle_state = wait_for_input(env, master_bot_left, master_bot_right)
        if BOTH_OPEN(handle_state):
            quit()

        """ 
        format of capture buffer is
        
        states  [s0, s1, s2, s3]
        actions [a0, a1, a2]
        
        s3 in this example is the terminal state and has no action associated with it
        
        each primitive run on the robot will append it's transitions to the capture buffer
        policy index maps actions to the policy in the sequence
        """
        states_all, actions_all, timings_all, policy_index_all = [], [], [], []
        policy_info = []

        for i, (act_policy, capture_policy, capture) in enumerate(zip(sequence, human_capture_sequence, capture_flags)):
            policy = capture_policy if capture else act_policy
            states_i, actions_i, timings_i, terminal_state_i = policy.initial_policy(env, initial_state, master_bot_left, master_bot_right)
            states, actions, timings, terminal_state = policy.execute_policy(env, terminal_state_i, master_bot_left, master_bot_right)

            states_all += states_i + states
            actions_all += actions_i + actions
            timings_all += timings_i + timings
            policy_index_all += [i * 2] * len(states_i) + [i * 2 + 1] * len(states)
            checkpoint_info = policy.checkpoint_info
            policy_info += [repr(policy)]

            handle_state = wait_for_input(env, master_bot_left, master_bot_right, block_until='any',
                                              message="CLOSE_RIGHT for PASS, CLOSE_LEFT for FAIL, OPEN_LEFT for SCRATCH")

            if RIGHT_HANDLE_CLOSED(handle_state):
                print("Saving PASS")
                episode_idx = get_auto_index(policy.dataset_dir)
                dataset_name = f'episode_{episode_idx}'
                episode_path = validate_dataset(policy.dataset_dir, dataset_name, overwrite=False)
                print(dataset_name + '\n')
                save_episode(episode_path, checkpoint_info.guid, policy.task['camera_names'], policy.task['episode_len'], states, actions,
                             terminal_state)
                update_panel(i, episode_path)
                checkpoint_info.values['trials'].append(1)
                checkpoint_info.save()
                capture_flags[i] = False

            elif LEFT_HANDLE_CLOSED(handle_state):
                print('FAIL')
                # capture fails in a subdirectory
                dataset_fail_dir = policy.dataset_dir + '/fails'
                episode_idx = get_auto_index(dataset_fail_dir)
                dataset_name = f'episode_{episode_idx}'
                episode_path = validate_dataset(dataset_fail_dir, dataset_name, overwrite=False)
                save_episode(episode_path, checkpoint_info.guid, policy.task['camera_names'], policy.task['episode_len'], states, actions,
                             terminal_state)

                # record the result
                checkpoint_info.values['trials'].append(0)
                checkpoint_info.save()
                update_panel(i, episode_path)
                capture_flags[i] = True
                break

            elif LEFT_HANDLE_OPEN(handle_state):
                print("SCRATCH")
                break

            elif RIGHT_HANDLE_OPEN(handle_state):
                print("QUIT")
                quit()

            initial_state = terminal_state

        print(f'trajectory states: {len(states_all)} actions: {len(actions_all)}')

        # if args.save_task is not None:
        #     task = TASK_CONFIGS[args.save_task]
        #     dataset_dir = task['dataset_dir']
        #     cam_names = task['cam_names']
        #     max_timesteps = task['episode_len']
        #     Path(dataset_dir).mkdir(exist_ok=True)
        #     episode_id = get_auto_index(dataset_dir)
        #     episode = f'{dataset_dir}/episode_{episode_id}'
        #     save_episode(episode, cam_names, max_timesteps, states, actions, policy_info=policy_info, policy_index=policy_index)

        reboot = args.reboot_every_episode


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--routine_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--reboot_every_episode', action='store_true', help='Do a full before each sequence',
                        default=False,
                        required=False)
    parser.add_argument('--current_limit', type=int, help='gripper current limit', default=300, required=False)
    parser.add_argument('--save_task', type=str, help='save trajectory as task', default=None, required=False)
    parser.add_argument('--capture_mode', type=str, help='save trajectory as task', default=None, required=False,
                        choices=['ALL', 'LAST'])

    # dummy arguments to make DETR happy
    parser.add_argument('--task_name', type=str, default='empty')
    parser.add_argument('--policy_class', type=str, default='ACT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default='dummy')

    args = parser.parse_args()

    if args.save_task is not None:
        assert args.save_task in TASK_CONFIGS, '--save_task {args.save_task} not found in TASK_CONFIG'

    main(args)
