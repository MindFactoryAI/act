from imitate_episodes import execute_policy_on_env, load_policy_and_stats, CheckPointInfo
from aloha_scripts.constants import TASK_CONFIGS, DT, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_CLOSE, \
    PUPPET_GRIPPER_JOINT_OPEN
from record_episodes import LEFT_HANDLE_CLOSED, RIGHT_HANDLE_CLOSED, LEFT_HANDLE_OPEN, RIGHT_HANDLE_OPEN
from aloha_scripts.robot_utils import get_arm_joint_positions, get_arm_gripper_positions, move_grippers, torque_on
from aloha_scripts.record_episodes import validate_dataset, teleoperate, print_dt_diagnosis, wait_for_input, \
    save_episode, get_auto_index
from pathlib import Path
import time
import numpy as np


def lerp_trajectory(bot, target_pose, num_steps):
    curr_pose = get_arm_joint_positions(bot)
    return np.linspace(curr_pose, target_pose, num_steps)


def get_gripper_position_normalized(master_bot):
    return MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot.dxl.joint_states.position[6])


class LinearMoveToStartPose:
    def __init__(self, task_name, move_time):
        task = TASK_CONFIGS[task_name]
        target_pose_left = task['start_left_arm_pose']
        target_pose_right = task['start_right_arm_pose']
        self.linear_move_arms = LinearMoveArms(target_pose_left, target_pose_right, move_time)

    def execute(self, env, states, actions, timings, master_bot_left=None, master_bot_right=None):
        return self.linear_move_arms.execute(env, states, actions, timings, master_bot_left, master_bot_right)


class LinearMoveArms:
    def __init__(self, target_pose_left, target_pose_right, move_time, gripper_target_left=None, gripper_target_right=None):
        self.target_pose_left = target_pose_left
        self.target_pose_right = target_pose_right
        self.move_time = move_time
        self.gripper_target_left = gripper_target_left
        self.gripper_target_right = gripper_target_right

    def execute(self, env, states, actions, timings, master_bot_left=None, master_bot_right=None):
        assert len(states) >= 1, "Please ensure there is at least an initial state provided to start"
        num_steps = int(self.move_time / DT)

        puppet_left_traj = lerp_trajectory(env.puppet_bot_left, self.target_pose_left, num_steps)
        puppet_right_traj = lerp_trajectory(env.puppet_bot_right, self.target_pose_right, num_steps)

        if self.gripper_target_left is not None:
            gripper_target_left = self.gripper_target_left
        elif len(actions) > 0:
            gripper_target_left = np.array([actions[-1][6]])
        else:
            gripper_target_left = np.array([PUPPET_GRIPPER_POSITION_CLOSE])

        if self.gripper_target_right is not None:
            gripper_target_right = self.gripper_target_right
        elif len(actions) > 0:
            gripper_target_right = np.array([actions[-1][7+6]])
        else:
            gripper_target_right = np.array([PUPPET_GRIPPER_POSITION_CLOSE])

        if master_bot_left:
            master_left_traj = lerp_trajectory(master_bot_left, self.target_pose_left, num_steps)
        if master_bot_right:
            master_right_traj = lerp_trajectory(master_bot_right, self.target_pose_right, num_steps)

        for t in range(num_steps):
            t0 = time.time()
            action = np.concatenate(
                [puppet_left_traj[t], gripper_target_left, puppet_right_traj[t], gripper_target_right]
            )
            t1 = time.time()
            state = env.step(action)
            t2 = time.time()
            states.append(state)
            actions.append(action)
            timings.append([t0, t1, t2])
            if master_bot_left:
                master_bot_left.arm.set_joint_positions(master_left_traj[t], blocking=False)
            if master_bot_right:
                master_bot_right.arm.set_joint_positions(master_right_traj[t], blocking=False)
            time.sleep(DT)
        return states, actions, timings, False


class ACTPrimitive:
    def __init__(self, task_name, checkpoint_path, chunk_size=100, hidden_dim=512, dim_feedforward=3200):
        self.task_name = task_name
        self.checkpoint_path = checkpoint_path
        self.task = TASK_CONFIGS[task_name]
        self.camera_names = self.task['camera_names']
        self.state_dim = 14
        self.evaluate = False
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

    def execute(self, env, states, actions, timings, master_bot_left=None, master_bot_right=None):
        assert len(states) >= 1, "Please ensure there is at least an initial state provided to start the policy"
        if self.policy is None:
            self.policy, self.stats = load_policy_and_stats(self.policy_config, self.checkpoint_path)
        states, actions, timings = \
            execute_policy_on_env(self.policy, env, states, actions, timings, self.task['episode_len'], self.state_dim, self.stats, self.camera_names,
                                  master_bot_left=master_bot_left, master_bot_right=master_bot_right)

        done = False
        if self.evaluate:
            checkpoint_info = CheckPointInfo.load(self.checkpoint_path)
            print("close left handle for fail, close right handle for pass, open left for scratch, open right for quit")
            handle_state = wait_for_input(env, master_bot_left, master_bot_right, block_until="any")
            if LEFT_HANDLE_CLOSED(handle_state):
                print("FAIL")
                checkpoint_info.values['trials'].append(0)
                checkpoint_info.update()
                done = True
            if RIGHT_HANDLE_CLOSED(handle_state):
                print("PASS")
                checkpoint_info.values['trials'].append(1)
                checkpoint_info.update()
            if LEFT_HANDLE_OPEN(handle_state):
                print("SCRATCH")
                done = True
            if RIGHT_HANDLE_OPEN(handle_state):
                exit()

        return states, actions, timings, done

class Capture:
    def __init__(self, task_name, episode_index=None):
        self.task = TASK_CONFIGS[task_name]
        self.dataset_dir = self.task['dataset_dir']
        assert Path(self.dataset_dir).exists(), f"dataset_dir: {self.dataset_dir} does not exist"

    def execute(self, env, states, actions, timings, master_bot_left, master_bot_right):
        assert len(states) >= 1, "Please ensure there is at least an initial state provided to start a recording"
        start_pos_states, start_pos_actions = len(states), len(actions)

        wait_for_input(env, master_bot_left, master_bot_right)

        timesteps, actions, actual_dt_history = \
            teleoperate(states, actions, timings, self.task['episode_len'], env, master_bot_left, master_bot_right)

        # Torque on both master bots
        torque_on(master_bot_left)
        torque_on(master_bot_right)

        # Open puppet grippers
        env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
        env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
        move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
        env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")

        freq_mean = print_dt_diagnosis(actual_dt_history[start_pos_actions:])
        if freq_mean < 42:
            raise Exception("Frequency Mean less than 42, robot capture is too slow")

        print("right_close for save, left_close for discard")
        handle_state = wait_for_input(env, master_bot_left, master_bot_right, block_until='any')

        if RIGHT_HANDLE_CLOSED(handle_state):
            episode_idx = get_auto_index(self.dataset_dir)
            dataset_name = f'episode_{episode_idx}'
            dataset_path = validate_dataset(self.dataset_dir, dataset_name, overwrite=False)
            print(dataset_name + '\n')

            # the initial state is at start_pos_state - 1, the first action taken is at start_pos_actions
            save_states, save_actions = states[start_pos_states-1:], actions[start_pos_actions:]
            did_save = save_episode(dataset_path, self.task['camera_names'], self.task['episode_len'], save_states, save_actions)

        return timesteps, actions, actual_dt_history, True

