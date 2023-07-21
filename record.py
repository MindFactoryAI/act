import argparse
from aloha_scripts.constants import DT, TASK_CONFIGS, START_ARM_POSE
from aloha_scripts.record_episodes import get_auto_index, capture_one_episode, opening_ceremony, wait_for_input
from aloha_scripts.robot_utils import move_arms
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from real_env import make_real_env, get_action
from imitate_episodes import execute_policy_on_env, load_policy_and_stats
from primitives import LinearMoveToStartPose, ACTPrimitive, Capture

ROUTINES = {
    'record_drop_battery_in_slot_only': [
        LinearMoveToStartPose('grasp_battery', move_time=1.0),
        ACTPrimitive('grasp_battery', '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt'),
        LinearMoveToStartPose('drop_battery_in_slot_only', move_time=1.0),
        Capture('drop_battery_in_slot_only')
    ],
    'drop_battery_in_slot': [
        LinearMoveToStartPose('grasp_battery', move_time=1.0),
        ACTPrimitive('grasp_battery', '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt'),
        LinearMoveToStartPose('drop_battery_in_slot_only', move_time=1.0),
        ACTPrimitive('drop_battery_in_slot_only', '/mnt/magneto/checkpoints/drop_battery_in_slot_only/noble-shape-2/policy_best_inv_learning_error_0.04085.ckpt')
    ],
    'record_push_battery_in_slot': [
        LinearMoveToStartPose('grasp_battery', move_time=1.0),
        ACTPrimitive('grasp_battery',
                     '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt'),
        LinearMoveToStartPose('drop_battery_in_slot_only', move_time=1.0),
        ACTPrimitive('drop_battery_in_slot_only',
                     '/mnt/magneto/checkpoints/drop_battery_in_slot_only/noble-shape-2/policy_best_inv_learning_error_0.04085.ckpt'),
        LinearMoveToStartPose('push_battery_in_slot', move_time=0.5),
        Capture('push_battery_in_slot')
    ],
}


def main(args):

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    reboot = True  # always reboot on the first episode

    sequence = ROUTINES[args.routine_name]

    for primitive in sequence:
        if hasattr(primitive, "evaluate"):
            primitive.evaluate = True

    policy_index = []

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
        states, actions, timings = [initial_state], [], []

        for i, policy in enumerate(sequence):
            states, actions, timings, done = policy.execute(env, states, actions, timings,  master_bot_left, master_bot_right)
            policy_index += [i] * len(actions)
            if done:
                break

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
