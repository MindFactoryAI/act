import argparse
from pathlib import Path

from aloha_scripts.constants import START_ARM_POSE, TASK_CONFIGS
from aloha_scripts.record_episodes import opening_ceremony
from data_utils import save_episode, validate_dataset, get_auto_index, Episode
from robot_utils import wait_for_input, LEFT_HANDLE_CLOSED, RIGHT_HANDLE_CLOSED, LEFT_HANDLE_OPEN, RIGHT_HANDLE_OPEN
from checkpoint import CheckPointInfo
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from real_env import make_real_env
from primitives import LinearMoveToStartPose, ACTPrimitive, Capture
from matplotlib import pyplot as plt
import numpy as np

ROUTINES = {
    'record_drop_battery_in_slot_only': {
        'program': [
            LinearMoveToStartPose('grasp_battery', move_time=1.0),
            ACTPrimitive('grasp_battery',
                         '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt'),
            LinearMoveToStartPose('drop_battery_in_slot_only', move_time=1.0),
            Capture('drop_battery_in_slot_only')
        ]},
    'drop_battery_in_slot': {
        'program': [
            LinearMoveToStartPose('grasp_battery', move_time=1.0),
            ACTPrimitive('grasp_battery'),
            LinearMoveToStartPose('drop_battery_in_slot_only', move_time=1.0),
            ACTPrimitive('drop_battery_in_slot_only')
        ]},
    'record_push_battery_in_slot': {
        'program': [
            LinearMoveToStartPose('grasp_battery', move_time=1.0),
            ACTPrimitive('grasp_battery',
                         '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt'),
            LinearMoveToStartPose('drop_battery_in_slot_only', move_time=1.0),
            ACTPrimitive('drop_battery_in_slot_only',
                         '/mnt/magneto/checkpoints/drop_battery_in_slot_only/noble-shape-2/policy_best_inv_learning_error_0.04085.ckpt'),
            LinearMoveToStartPose('push_battery_in_slot', move_time=0.5),
            Capture('push_battery_in_slot')
        ]},
    'record_push_battery_in_slot_only': {
        'program': [
            LinearMoveToStartPose('push_battery_in_slot', move_time=0.5),
            Capture('push_battery_in_slot')
        ]},
    'push_battery_in_slot_only': {
        'program': [
            LinearMoveToStartPose('push_battery_in_slot', move_time=0.5),
            ACTPrimitive('push_battery_in_slot',
                         '/mnt/magneto/checkpoints/push_battery_in_slot/peach-disco-3/policy_best_inv_learning_error_0.05223.ckpt'
                         # '/mnt/magneto/checkpoints/push_battery_in_slot/peach-disco-3/policy_min_val_loss0.33072.ckpt'
                         # '/mnt/magneto/checkpoints/push_battery_in_slot/dummy-osa5na9u/policy_min_val_loss0.31343.ckpt'
                         )
        ]},
    'slot_battery': {
        'program': [
            LinearMoveToStartPose('grasp_battery', move_time=1.0),
            ACTPrimitive('grasp_battery',
                         '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt'),
            LinearMoveToStartPose('drop_battery_in_slot_only', move_time=1.0),
            ACTPrimitive('drop_battery_in_slot_only',
                         '/mnt/magneto/checkpoints/drop_battery_in_slot_only/noble-shape-2/policy_best_inv_learning_error_0.04085.ckpt'),
            LinearMoveToStartPose('push_battery_in_slot', move_time=0.5),
            ACTPrimitive('push_battery_in_slot',
                         '/mnt/magneto/checkpoints/push_battery_in_slot/dummy-osa5na9u/policy_min_val_loss0.31343.ckpt')
        ]},
}


def save_result(task_name, checkpoint_path, states, actions, terminal_state, reward, policy_info):
    task = TASK_CONFIGS[task_name]
    checkpoint_info = CheckPointInfo.load(checkpoint_path)
    results_dir = Path(checkpoint_path).parent / Path(checkpoint_path).stem
    results_dir.mkdir(exist_ok=True)
    next_trial = len(checkpoint_info.values['trials'])
    dataset_filepath = str(results_dir / Path(f'episode_{next_trial}'))
    checkpoint_info.values['trials'].append(reward)
    checkpoint_info.update()
    save_episode(dataset_filepath, task['camera_names'], task['episode_len'], states, actions,
                 terminal_state=terminal_state, result=reward, policy_info=policy_info)
    return dataset_filepath


def main(args):
    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    reboot = True  # always reboot on the first episode

    sequence = ROUTINES[args.routine_name]['program']

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(8, 12), dpi=80)
    ax_img = ax.imshow(np.zeros((640, 480)))
    fig.tight_layout()
    fig.canvas.draw()
    plt.draw()
    plt.pause(1)

    def update_panel(episode_path):
        frame = Episode(f'{episode_path}.hdf5').get_frame(0, "RGB")
        cam_low = Episode(f'{episode_path}.hdf5').split_frame(frame)['cam_low']
        ax_img.set_data(np.fliplr(cam_low.swapaxes(0, 1)))
        plt.pause(1)

    while True:

        opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, reboot,
                         args.current_limit, START_ARM_POSE[:6], START_ARM_POSE[:6])

        initial_state = env.reset()

        wait_for_input(env, master_bot_left, master_bot_right)

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

        for i, policy in enumerate(sequence):
            states, actions, timings, terminal_state = policy.execute(env, initial_state, master_bot_left,
                                                                      master_bot_right)

            states_all += states
            actions_all += actions
            timings_all += timings
            policy_index_all += [i] * len(states)
            policy_info += [repr(policy)]

            if policy.evaluate:
                handle_state = wait_for_input(env, master_bot_left, master_bot_right, block_until="any",
                                              message="close left handle for fail, close right handle for pass, open left for scratch, open right for quit")

                checkpoint_info = CheckPointInfo.load(policy.checkpoint_path)
                if LEFT_HANDLE_CLOSED(handle_state):
                    print('FAIL')
                    episode_path = save_result(policy.task_name, policy.checkpoint_path, states, actions,
                                               terminal_state,
                                               reward=0, policy_info=policy_info)
                    checkpoint_info.values['trials'].append(0)
                    checkpoint_info.update()
                    update_panel(episode_path)
                    break
                elif RIGHT_HANDLE_CLOSED(handle_state):
                    print('PASS')
                    episode_path = save_result(policy.task_name, policy.checkpoint_path, states, actions,
                                               terminal_state,
                                               reward=1, policy_info=policy_info)
                    checkpoint_info.values['trials'].append(1)
                    checkpoint_info.update()
                    update_panel(episode_path)
                elif LEFT_HANDLE_OPEN(handle_state):
                    print("SCRATCH")
                    break
                elif RIGHT_HANDLE_OPEN(handle_state):
                    print("QUIT")
                    quit()

            if isinstance(policy, Capture):
                handle_state = wait_for_input(env, master_bot_left, master_bot_right, block_until='any',
                                              message="right_close for save, left close for discard")

                if RIGHT_HANDLE_CLOSED(handle_state):
                    print("Saving PASS")
                    episode_idx = get_auto_index(policy.dataset_dir)
                    dataset_name = f'episode_{episode_idx}'
                    episode_path = validate_dataset(policy.dataset_dir, dataset_name, overwrite=False)
                    print(dataset_name + '\n')
                    save_episode(episode_path, policy.task['camera_names'], policy.task['episode_len'], states, actions,
                                 terminal_state)
                    update_panel(episode_path)

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
