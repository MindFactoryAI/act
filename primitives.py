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
from checkpoint import CheckPointInfo

CHECKPOINT_DIR = '/mnt/magneto/checkpoints'

CANONICAL_CHECKPOINTS = {
    'grasp_battery': f'/mnt/magneto/checkpoints/grasp_battery/still-music-23/policy_min_val_loss0.17796.ckpt',
    'drop_battery_in_slot_only': f'{CHECKPOINT_DIR}/drop_battery_in_slot_only/rare-serenity-3/policy_best_inv_learning_error_0.04715.ckpt',
    'push_battery_in_slot': f'{CHECKPOINT_DIR}/push_battery_in_slot/dummy-osa5na9u/policy_min_val_loss0.31343.ckpt',
    'slot_battery_novar': None,
}


def lerp_trajectory(bot, target_pose, num_steps):
    curr_pose = get_arm_joint_positions(bot)
    return np.linspace(curr_pose, target_pose, num_steps)


def get_gripper_position_normalized(master_bot):
    return MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot.dxl.joint_states.position[6])


class Primitive:
    def __init__(self):
        self.task_name = None
        self.initial_policy = None
        self.execute_policy = None

    def initial(self, env, initial_state, master_bot_left=None, master_bot_right=None):
        self.initial_policy(env, initial_state, master_bot_left, master_bot_right)

    def execute(self, env, initial_state, master_bot_left=None, master_bot_right=None):
        self.execute_policy(env, initial_state, master_bot_left, master_bot_right)

    def __repr__(self):
        return f"initial_policy: {self.initial_policy}, execute_policy: {self.execute_policy}"


class ACTPrimitive(Primitive):
    def __init__(self, task_name, checkpoint_path=None, rollout_len_override=None):
        super().__init__()
        self.task_name = task_name
        self.task = TASK_CONFIGS[task_name]
        self.initial_policy = LerpJointPosPolicy(task_name)
        self.execute_policy = ACTPolicy(task_name, checkpoint_path, rollout_len_override=rollout_len_override)
        self.dataset_dir = TASK_CONFIGS[task_name]['dataset_dir']

    @property
    def checkpoint_info(self):
        return CheckPointInfo.load(self.execute_policy.checkpoint_path)


class CapturePrimitive(Primitive):
    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name
        self.task = TASK_CONFIGS[task_name]
        self.initial_policy = LerpJointPosPolicy(task_name)
        self.execute_policy = CapturePolicy(task_name)
        self.dataset_dir = TASK_CONFIGS[task_name]['dataset_dir']

    @property
    def checkpoint_info(self):
        checkpoint_path = Path(self.task['dataset_dir'])/Path('human.ckpt')
        return CheckPointInfo.load(checkpoint_path, human=True)


class LerpJointPosPolicy:
    def __init__(self, task_name, initial_move_time=None, initial_gripper_targets=None):
        self.task_name = task_name
        self.task = TASK_CONFIGS[task_name]
        self.target_pose_left = self.task['start_left_arm_pose']
        self.target_pose_right = self.task['start_right_arm_pose']
        if initial_move_time is not None:
            self.move_time = initial_move_time
        elif 'initial_move_time' in self.task:
            self.move_time = TASK_CONFIGS['initial_move_time']
        else:
            self.move_time = 1.
        self.initial_gripper_targets = initial_gripper_targets

    def __call__(self, env, initial_state, master_bot_left=None, master_bot_right=None):
        num_steps = int(self.move_time / DT)
        states = []
        actions = []
        timings = []

        puppet_left_traj = lerp_trajectory(env.puppet_bot_left, self.target_pose_left, num_steps)
        puppet_right_traj = lerp_trajectory(env.puppet_bot_right, self.target_pose_right, num_steps)

        if self.initial_gripper_targets is not None:
            gripper_target_left = self.initial_gripper_targets[0]
            gripper_target_right = self.initial_gripper_targets[1]
        else:
            gripper_target_left = np.array([get_gripper_position_normalized(master_bot_left)])
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
        return f"initial_pose_left: {self.target_pose_left} " \
               f"initial_pose_right {self.target_pose_right}" \
               f"initial_gripper_targets: {self.initial_gripper_targets} " \
               f"initial_move_time: {self.move_time}"


class ACTPolicy:
    def __init__(self, task_name, checkpoint_path=None,
                 chunk_size=100, hidden_dim=512, dim_feedforward=3200, rollout_len_override=None):
        """

        @param task_name: name of the task from aloha_scripts.constants TASK_CONFIG
        @param initial_move_time: time in seconds to move to start position, if not set will default to initial_move_time in TASK_CONFIG 1. second otherwise
        @param initial_gripper_targets: [left, right], if none will just be the current settings
        @param checkpoint_path: path to checkpoint, if None will take the checkpoint from CANONICAL_CHECKPOINTS['task_name']
        @param chunk_size: number of actions to lookahead and average over (k from the paper)
        @param hidden_dim: of the embedding in the transformer
        @param dim_feedforward:
        @param evaluate: evaluate if checkpoint is set
        """

        self.task_name = task_name
        self.task = TASK_CONFIGS[task_name]
        self.dataset_dir = self.task['dataset_dir']
        self.episode_len = self.task['episode_len'] if rollout_len_override is None else rollout_len_override
        canonical_checkpint = CANONICAL_CHECKPOINTS[task_name] if task_name in CANONICAL_CHECKPOINTS else None
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else canonical_checkpint
        self.camera_names = self.task['camera_names']
        self.state_dim = 14
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

    def __call__(self, env, initial_state, master_bot_left=None, master_bot_right=None):

        if self.policy is None:  # lazy load the policy weights
            self.policy, self.stats = load_policy_and_stats(self.policy_config, self.checkpoint_path)

        states, actions, timings, terminal_state = \
            execute_policy_on_env(self.policy, env, initial_state, self.episode_len, self.state_dim, self.stats,
                                  self.camera_names,
                                  master_bot_left=master_bot_left, master_bot_right=master_bot_right)

        return states, actions, timings, terminal_state

    def __repr__(self):
        return f"{self.__class__} task_name: {self.task_name} " \
               f"checkpoint: {self.checkpoint_path}             " \
               f"policy_config : {self.policy_config}"


class CapturePolicy:
    def __init__(self, task_name):

        self.task_name = task_name
        self.task = TASK_CONFIGS[task_name]
        self.episode_len = self.task['episode_len']
        self.dataset_dir = self.task['dataset_dir']

    def __call__(self, env, initial_state, master_bot_left, master_bot_right, open_grippers_after=False):
        wait_for_input(env, master_bot_left, master_bot_right)

        states, actions, timings = \
            teleoperate([initial_state], [], [], self.episode_len, env, master_bot_left, master_bot_right)
        terminal_state = states[-1]
        states = states[:-1]

        # Torque on both master bots
        torque_on(master_bot_left)
        torque_on(master_bot_right)

        if open_grippers_after:
            env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
            env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                          move_time=0.5)
            env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
            env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")

        freq_mean = print_dt_diagnosis(timings)
        if freq_mean < 42:
            raise Exception("Frequency Mean less than 42, robot capture is too slow")

        return states, actions, timings, terminal_state

    def __repr__(self):
        return f"{self.__class__} task_name: {self.task_name} CAPTURED_BY_HUMAN"
