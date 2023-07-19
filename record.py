import argparse
from aloha_scripts.constants import DT, TASK_CONFIGS, get_start_arm_pose
from aloha_scripts.record_episodes import get_auto_index, capture_one_episode, opening_ceremony, wait_for_start
from aloha_scripts.robot_utils import move_arms
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from real_env import make_real_env, get_action
from imitate_episodes import execute_policy_on_env, load_policy_and_stats
from primitives import build_primitive, PRIMITIVES


def main(args):
    current_limit = args['current_limit']

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False, task=args['task_name'])

    reboot = True  # always reboot on the first episode

    start_left_arm_pose, start_right_arm_pose = get_start_arm_pose(args['task_name'])
    grasp_battery = build_primitive(PRIMITIVES['grasp_battery'])
    move_to_start = build_primitive(PRIMITIVES['move_arms_to_start_pose'])
    capture_drop_battery_in_slot_only = build_primitive(PRIMITIVES['capture_drop_battery_in_slot_only'])

    while True:

        opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, reboot,
                         current_limit, start_left_arm_pose, start_right_arm_pose)

        initial_state = env.reset()

        # execute first policy
        wait_for_start(master_bot_left, master_bot_right, human_takeover=False)

        state, actions, timings = [initial_state], [], []
        state, actions, timings = grasp_battery.execute(env, state, actions, timings,  master_bot_left, master_bot_right)
        state, actions, timings = move_to_start.execute(env, state, actions, timings, master_bot_left, master_bot_right)
        state, actions, timings = capture_drop_battery_in_slot_only.execute(env, state, actions, timings, master_bot_left, master_bot_right)

        reboot = args['reboot_every_episode']


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true', default=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--reboot_every_episode', action='store_true', help='Episode index.', default=False,
                        required=False)
    parser.add_argument('--current_limit', type=int, help='gripper current limit', default=300, required=False)

    # dummy arguments to make DETR happy
    parser.add_argument('--policy_class', type=str, default='ACT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default='dummy')

    main(vars(parser.parse_args()))
    # debug()
