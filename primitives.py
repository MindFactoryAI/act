from imitate_episodes import execute_policy_on_env, load_policy_and_stats
from checkpoint import CheckPointInfo
from aloha_scripts.constants import TASK_CONFIGS, DT, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_CLOSE, \
    PUPPET_GRIPPER_JOINT_OPEN
from aloha_scripts.robot_utils import get_arm_joint_positions, get_arm_gripper_positions, move_grippers, torque_on
from aloha_scripts.record_episodes import teleoperate, print_dt_diagnosis
from data_utils import save_episode, get_auto_index
from robot_utils import wait_for_input, LEFT_HANDLE_CLOSED, RIGHT_HANDLE_CLOSED, LEFT_HANDLE_OPEN, RIGHT_HANDLE_OPEN
from pathlib import Path
import time
import numpy as np


CHECKPOINT_DIR = '/mnt/magneto/checkpoints'

CANONICAL_CHECKPOINTS = {
    'grasp_battery': f'{CHECKPOINT_DIR}/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt',
    'drop_battery_in_slot_only': f'{CHECKPOINT_DIR}/drop_battery_in_slot_only/rare-serenity-3/policy_best_inv_learning_error_0.04715.ckpt',
    'push_battery_in_slot': f'{CHECKPOINT_DIR}/push_battery_in_slot/dummy-osa5na9u/policy_min_val_loss0.31343.ckpt'
}


def lerp_trajectory(bot, target_pose, num_steps):
    curr_pose = get_arm_joint_positions(bot)
    return np.linspace(curr_pose, target_pose, num_steps)


def get_gripper_position_normalized(master_bot):
    return MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot.dxl.joint_states.position[6])


class LinearMoveToStartPose:
    def __init__(self, task_name, move_time):
        self.task_name = task_name
        task = TASK_CONFIGS[task_name]
        target_pose_left = task['start_left_arm_pose']
        target_pose_right = task['start_right_arm_pose']
        self.linear_move_arms = LinearMoveArms(target_pose_left, target_pose_right, move_time)
        self.evaluate = False

    def execute(self, env, initial_state, master_bot_left=None, master_bot_right=None):
        return self.linear_move_arms.execute(env, initial_state, master_bot_left, master_bot_right)

    def __repr__(self):
        return \
            f"{self.__class__} " \
            f"task: {self.task_name} " \
            f"target_pose_left: {self.linear_move_arms.target_pose_left} " \
            f"target_pose_right {self.linear_move_arms.target_pose_right}" \
            f"gripper_target_left: {self.linear_move_arms.gripper_target_left} " \
            f"gripper_target_right: {self.linear_move_arms.gripper_target_right} " \
            f"move_time: {self.linear_move_arms.move_time}"


class LinearMoveArms:
    def __init__(self, target_pose_left, target_pose_right, move_time, gripper_target_left=None, gripper_target_right=None):
        self.target_pose_left = target_pose_left
        self.target_pose_right = target_pose_right
        self.move_time = move_time
        self.gripper_target_left = gripper_target_left
        self.gripper_target_right = gripper_target_right
        self.evaluate = False

    def execute(self, env, initial_state, master_bot_left=None, master_bot_right=None):
        num_steps = int(self.move_time / DT)
        states = []
        actions = []
        timings = []

        puppet_left_traj = lerp_trajectory(env.puppet_bot_left, self.target_pose_left, num_steps)
        puppet_right_traj = lerp_trajectory(env.puppet_bot_right, self.target_pose_right, num_steps)

        if self.gripper_target_left is not None:
            gripper_target_left = self.gripper_target_left
        else:
            gripper_target_left = np.array([get_gripper_position_normalized(master_bot_left)])

        if self.gripper_target_right is not None:
            gripper_target_right = self.gripper_target_right
        else:
            gripper_target_right = np.array([get_gripper_position_normalized(master_bot_right)])

        if master_bot_left:
            master_left_traj = lerp_trajectory(master_bot_left, self.target_pose_left, num_steps)
        if master_bot_right:
            master_right_traj = lerp_trajectory(master_bot_right, self.target_pose_right, num_steps)

        state = initial_state

        for t in range(num_steps):
            t0 = time.time()
            action = np.concatenate(
                [puppet_left_traj[t], gripper_target_left, puppet_right_traj[t], gripper_target_right]
            )

            states.append(state)
            actions.append(action)

            t1 = time.time()
            next_state = env.step(action)
            t2 = time.time()

            timings.append([t0, t1, t2])

            if master_bot_left:
                master_bot_left.arm.set_joint_positions(master_left_traj[t], blocking=False)
            if master_bot_right:
                master_bot_right.arm.set_joint_positions(master_right_traj[t], blocking=False)
            time.sleep(DT)
            state = next_state

        return states, actions, timings, state

    def __repr__(self):
        return \
            f"{self.__class__} " \
            f"target_pose_left: {self.target_pose_left} " \
            f"target_pose_right {self.target_pose_right}" \
            f"gripper_target_left: {self.gripper_target_left} " \
            f"gripper_target_right: {self.gripper_target_right} " \
            f"move_time: {self.move_time}"


class ACTPrimitive:
    def __init__(self, task_name, checkpoint_path=None, chunk_size=100, hidden_dim=512, dim_feedforward=3200, evaluate=True):
        self.task_name = task_name
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else CANONICAL_CHECKPOINTS[task_name]
        self.task = TASK_CONFIGS[task_name]
        self.camera_names = self.task['camera_names']
        self.state_dim = 14
        self.evaluate = evaluate
        self.policy_config = {
            'lr': 1e-5,
            'num_queries': chunk_size,
            'kl_weight': 0.0,
            'hidden_dim': hidden_dim,
            'dim_feedforward': dim_feedforward,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': self.camera_names
        }
        self.policy, self.stats = None, None

    def execute(self, env, initial_state, master_bot_left=None, master_bot_right=None):

        if self.policy is None:  # lazy load the policy weights
            self.policy, self.stats = load_policy_and_stats(self.policy_config, self.checkpoint_path)

        states, actions, timings, terminal_state = \
            execute_policy_on_env(self.policy, env, initial_state, self.task['episode_len'], self.state_dim, self.stats, self.camera_names,
                                  master_bot_left=master_bot_left, master_bot_right=master_bot_right)

        return states, actions, timings, terminal_state

    def __repr__(self):
        return f"{self.__class__} task_name: {self.task_name} checkpoint: {self.checkpoint_path}"


class Capture:
    def __init__(self, task_name, episode_index=None):
        self.task_name = task_name
        self.task = TASK_CONFIGS[task_name]
        self.dataset_dir = self.task['dataset_dir']
        self.evaluate = False
        assert Path(self.dataset_dir).exists(), f"dataset_dir: {self.dataset_dir} does not exist"

    def execute(self, env, initial_state, master_bot_left, master_bot_right):

        wait_for_input(env, master_bot_left, master_bot_right)

        timesteps, actions, actual_dt_history = \
            teleoperate([initial_state], [], [], self.task['episode_len'], env, master_bot_left, master_bot_right)
        terminal_state = timesteps[-1]
        timesteps = timesteps[:-1]

        # Torque on both master bots
        torque_on(master_bot_left)
        torque_on(master_bot_right)

        # Open puppet grippers
        env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
        env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
        move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
        env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")

        freq_mean = print_dt_diagnosis(actual_dt_history)
        if freq_mean < 42:
            raise Exception("Frequency Mean less than 42, robot capture is too slow")


        return timesteps, actions, actual_dt_history, terminal_state

    def __repr__(self):
        return f"{self.__class__} task_name: {self.task_name}"
