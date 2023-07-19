from imitate_episodes import execute_policy_on_env, load_policy_and_stats
from aloha_scripts.constants import TASK_CONFIGS, DT, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_CLOSE
from aloha_scripts.robot_utils import get_arm_joint_positions, get_arm_gripper_positions
import time
import numpy as np


PRIMITIVES = {
    "move_arms_to_start_pose": {
        "module_name": "primitives",
        "object_name": "LinearMoveArms",
        "args": [],
        "kwargs": {
            "target_pose_left": [0.2208932340145111, -0.37889325618743896, 1.2686021327972412,
                                 0.44025251269340515, -0.6135923266410828, -0.2178252786397934],
            "target_pose_right": [-0.14726215600967407, -0.5599030256271362, 1.3023496866226196,
                                  -0.058291271328926086, -0.3436117172241211, 0.02147573232650757],
            "move_time": 1.0,
        }
    },
    "grasp_battery": {
        "module_name": "primitives",
        "object_name": "ACTPrimitive",
        "args": [],
        "kwargs": {
            'task_name': 'grasp_battery',
            'checkpoint': '/mnt/magneto/checkpoints/grasp_battery/fancy-cherry-9/policy_best_inv_learning_error_0.05250.ckpt',
            'chunk_size': 100,
            'hidden_dim': 512,
            'dim_feedforward': 3200
        }
    }
}


def build_primitive(definition):
    module = __import__(definition['module_name'])
    class_ = getattr(module, definition['object_name'])
    return class_(*definition['args'], **definition['kwargs'])


def lerp_trajectory(bot, target_pose, num_steps):
    curr_pose = get_arm_joint_positions(bot)
    return np.linspace(curr_pose, target_pose, num_steps)


def get_gripper_position_normalized(master_bot):
    return MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot.dxl.joint_states.position[6])


class LinearMoveArms:
    def __init__(self, target_pose_left, target_pose_right, move_time, gripper_target_left=None, gripper_target_right=None):
        self.target_pose_left = target_pose_left
        self.target_pose_right = target_pose_right
        self.move_time = move_time
        self.gripper_target_left = gripper_target_left
        self.gripper_target_right = gripper_target_right

    def execute(self, env, states, actions, timings, master_bot_left=None, master_bot_right=None):
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
        return states, actions, timings


class ACTPrimitive:
    def __init__(self, task_name, checkpoint, chunk_size, hidden_dim, dim_feedforward):
        self.task_name = task_name
        self.task = TASK_CONFIGS[task_name]
        self.camera_names = self.task['camera_names']
        self.state_dim = 14
        policy_config = {
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
        self.policy, self.stats = load_policy_and_stats(policy_config, checkpoint)

    def execute(self, env, states, actions, timings, master_bot_left=None, master_bot_right=None):
        states, actions, timings = \
            execute_policy_on_env(self.policy, env, states, actions, timings, self.task['episode_len'], self.state_dim, self.stats, self.camera_names,
                                  master_bot_left=master_bot_left, master_bot_right=master_bot_right)
        return states, actions, timings
