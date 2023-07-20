import argparse
from aloha_scripts.constants import DT, TASK_CONFIGS, START_ARM_POSE
from aloha_scripts.record_episodes import get_auto_index, capture_one_episode, opening_ceremony, wait_for_start
from aloha_scripts.robot_utils import move_arms
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from real_env import make_real_env, get_action
from imitate_episodes import execute_policy_on_env, load_policy_and_stats
from primitives import build_primitive, PRIMITIVES

ROUTINES = {
    'record_drop_battery_in_slot_only': [
        PRIMITIVES['move_arms_to_grasp_battery_start_pose'],
        PRIMITIVES['grasp_battery'],
        PRIMITIVES['move_arms_to_drop_battery_in_slot_start_pose'],
        PRIMITIVES['capture_drop_battery_in_slot_only']
    ],
    'drop_battery_in_slot': [
        PRIMITIVES['move_arms_to_grasp_battery_start_pose'],
        PRIMITIVES['grasp_battery'],
        PRIMITIVES['move_arms_to_drop_battery_in_slot_start_pose'],
        PRIMITIVES['drop_battery_in_slot_only']
    ]
}


def main(args):

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    reboot = True  # always reboot on the first episode

    sequence = [build_primitive(primitive) for primitive in ROUTINES[args.routine_name]]

    while True:

        opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right, reboot,
                         args.current_limit, START_ARM_POSE[:6], START_ARM_POSE[:6])

        initial_state = env.reset()

        wait_for_start(env, master_bot_left, master_bot_right)

        """ 
        format of capture buffer is
        
        states  [s0, s1, s2, s3]
        actions [a0, a1, a2]
        
        s3 in this example is the terminal state and has no action associated with it
        
        each primitive run on the robot will append it's transitions to the capture buffer
        mark and slice the buffer accordingly to recover segments
        """
        states, actions, timings = [initial_state], [], []

        for policy in sequence:
            states, actions, timings = policy.execute(env, states, actions, timings,  master_bot_left, master_bot_right)

        reboot = args.reboot_every_episode


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--routine_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--reboot_every_episode', action='store_true', help='Do a full before each sequence', default=False,
                        required=False)
    parser.add_argument('--current_limit', type=int, help='gripper current limit', default=300, required=False)

    # dummy arguments to make DETR happy
    parser.add_argument('--task_name', type=str, default='empty')
    parser.add_argument('--policy_class', type=str, default='ACT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default='dummy')

    main(parser.parse_args())
    # debug()
