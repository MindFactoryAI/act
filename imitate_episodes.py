import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import sys
import tty
import termios
from pathlib import Path
import json

import wandb

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from constants import MASTER_GRIPPER_JOINT_UNNORMALIZE_FN
from interbotix_xs_msgs.msg import JointSingleCommand
from robot_utils import torque_on
from sim_env import BOX_POSE


import IPython
import time

e = IPython.embed


def trajectory_similarity(l1_dist, policy_actions, expert_dist):

    # KL divergence between distributions
    policy_dist = torch.distributions.Normal(policy_actions, torch.std(policy_actions))
    kl_div = torch.distributions.kl_divergence(policy_dist, expert_dist)

    # Trajectory Similarity Index
    tsi = l1_dist + kl_div

    return tsi


def best_checkpoint(checkpoint_dir, n=0):
    """
    checkpoint dir: directory to find checkpoints
    n: 0 returns best, 1, returns 2nd best, etc
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    if len(checkpoints) > 0:
        return str(checkpoints[n])

    policy_best = Path(checkpoint_dir)/Path('policy_best.ckpt')
    if policy_best.exists():
        return str(policy_best)

    return None


def find_checkpoint(ckpt_dir):
    checkpoints = list(Path(ckpt_dir).glob('*.ckpt'))
    for ckpt in checkpoints:
        if ckpt.name == 'policy_last.ckpt':
            return str(ckpt)

    policy_best = best_checkpoint(ckpt_dir, 0)
    if policy_best is not None:
        return policy_best

    else:
        print(f"No checkpoints found in {ckpt_dir}")
        exit(1)


def list_checkpoints(checkpoint_dir, prefix=None):
    if prefix is None:
        return sorted(list(Path(checkpoint_dir).glob('*.ckpt')))
    else:
        return sorted(list(Path(checkpoint_dir).glob(f'{prefix}*.ckpt')))


def save_best_checkpoints(checkpoint_dir, run,
                          val_loss, min_val_loss,
                          inv_learning_error, min_inv_learning_error,
                          policy, optimizer,
                          epoch_train_loss, epoch, top_n_checkpoints=5):
    """
    checkpoint_dir: the directory to save checkpoints
    run: wandb_run
    val_loss: the val loss
    min_val_loss: the minimum val_loss reached so far
    policy: the model
    optimizer: the optimizer
    epoch_train_loss: the training loss reached
    top_n_checkpoints: keep the best n checkpoints
    """

    if val_loss < min_val_loss:
        # Update the best validation loss
        min_val_loss = val_loss

        # Remove any existing checkpoints if there are more than n
        checkpoints = list_checkpoints(checkpoint_dir, 'policy_min_val_loss')
        if len(checkpoints) >= top_n_checkpoints:
            lowest_val_checkpoint = checkpoints[-1]
            if not Path(str(lowest_val_checkpoint) + '.data').exists():
                os.remove(os.path.join(checkpoints[-1]))

        # Save the new checkpoint
        torch.save({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": min_val_loss,
            "min_inv_learning_error": min_inv_learning_error,
            "model_state_dict": policy.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "wandb_id": run.id
        }, f'{checkpoint_dir}/policy_min_val_loss{val_loss:.5f}.ckpt')

    if inv_learning_error < min_inv_learning_error:
        # Update the best validation loss
        min_inv_learning_error = inv_learning_error

        # Remove any existing checkpoints if there are more than n
        checkpoints = list_checkpoints(checkpoint_dir, 'policy_best_inv_learning_error')
        if len(checkpoints) >= top_n_checkpoints:
            lowest_val_checkpoint = checkpoints[-1]
            if not Path(str(lowest_val_checkpoint) + '.data').exists():
                os.remove(os.path.join(checkpoints[-1]))

        # Save the new checkpoint
        torch.save({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": min_val_loss,
            "min_inv_learning_error": min_inv_learning_error,
            "model_state_dict": policy.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "wandb_id": run.id
        }, f'{checkpoint_dir}/policy_best_inv_learning_error_{inv_learning_error:.5f}.ckpt')

    return min_val_loss, min_inv_learning_error


class CheckPointInfo(object):
    def __init__(self, ckpt_path, values):
        self.ckpt_path = ckpt_path
        self.values = values

    @property
    def epoch(self):
        if 'epoch' in self.values:
            return self.values['epoch']
        else:
            return 0

    @property
    def val_loss(self):
        return self.values['val_loss']

    @property
    def train_loss(self):
        return self.values['train_loss']

    @property
    def sidecar(self):
        return Path(f'{str(Path(self.ckpt_path))}.data')

    @staticmethod
    def load(ckpt_path):
        checkpoint = Path(ckpt_path)
        if not checkpoint.exists():
            raise Exception(f'File {checkpoint} not found')

        sidecar = Path(f'{str(Path(ckpt_path))}.data')

        if not sidecar.exists():
            values = {
                'trials': [],
            }
            with open(sidecar, 'w') as file:
                json.dump(values, file)
        else:
            with open(sidecar, 'r') as file:
                values = json.load(file)

        ckpt_info = CheckPointInfo(ckpt_path, values)

        if 'val_loss' not in ckpt_info.values:
            try:
                checkpoint = torch.load(ckpt_path)
                for key in ['val_loss', 'train_loss', 'epoch']:
                    if key in checkpoint:
                        if isinstance(checkpoint[key], torch.Tensor):
                            value = checkpoint[key].item()
                        else:
                            value = str(checkpoint[key])
                        ckpt_info.values[key] = value
                    else:  # its the old format, so kludge it and move on
                        ckpt_info.values['val_loss'] = 10.
                ckpt_info.update()
            except RuntimeError:
                print(f'{ckpt_path} could not be loaded - the file is probably corrupt, delete it')

        return ckpt_info

    def update(self):
        with open(self.sidecar, 'w') as file:
            json.dump(self.values, file)

    @property
    def trials_n(self):
        return len(self.values['trials'])

    @property
    def successes(self):
        return sum(self.values['trials'])

    @property
    def success_rate(self):
        return self.successes / self.trials_n


def is_sim(task_name):
    return task_name[:4] == 'sim_'


def get_task_config(task_name):
    if is_sim(task_name):
        from constants import SIM_TASK_CONFIGS
        return SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        return TASK_CONFIGS[task_name]


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    task_config = get_task_config(task_name)
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    run_dir = None

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim(task_name),
        'resume': args['resume']
    }

    if is_eval:
        def select_run(ckpt_dir):
            run_dirs = [d for d in Path(ckpt_dir).iterdir() if d.is_dir()]
            for i, run in enumerate(run_dirs):
                print(f'{i} {run}')
            while True:
                selection = int(input('Enter a run: '))
                if selection < len(run_dirs):
                    return f'{run_dirs[selection]}'

        def select_checkpoint(run_dir):
            print(run_dir)
            checkpoints = list_checkpoints(run_dir)
            for i, checkpoint in enumerate(checkpoints):
                checkpoint_info = CheckPointInfo.load(checkpoint)
                print(f'{i}. {checkpoint} {checkpoint_info.epoch} {checkpoint_info.successes}/{checkpoint_info.trials_n}')
            while True:
                selection = int(input('Enter a checkpoint id: '))
                if selection < len(checkpoints):
                    return checkpoints[selection]

        def num_rollouts():
            while True:
                txt = input('Number of rollouts (Enter for default 5): ')
                if txt == '':
                    return 5
                else:
                    try:
                        return int(txt)
                    except ValueError:
                        continue

        run_dir = select_run(ckpt_dir)
        ckpt = select_checkpoint(run_dir)
        num_rollouts = num_rollouts()

        results = []

        success_rate, avg_return = eval_bc(config, ckpt, num_rollouts, save_episode=False)
        results.append([args['ckpt_path'], success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_bc(config)


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char


def execute_policy_on_env(policy, env, states, actions, actual_dt_history, max_timesteps, state_dim, stats, camera_names,
                          pre_process=None, post_process=None, num_queries=100, query_frequency=20,
                          master_bot_left=None, master_bot_right=None):

    if pre_process is None:
        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if post_process is None:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']# if post_process  is None else post_process

    all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()
    qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()

    if master_bot_left and master_bot_right:
        torque_on(master_bot_left)
        torque_on(master_bot_right)

    with torch.inference_mode():
        for t in range(max_timesteps):

            t0 = time.time()

            ### process previous timestep to get qpos and image_list
            state = states[-1]
            obs = state.observation

            qpos_numpy = np.array(obs['qpos'])
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_history[:, t] = qpos
            curr_image = get_image(state, camera_names)

            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)

            all_time_actions[[t], t:t + num_queries] = all_actions
            actions_for_curr_step = all_time_actions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action

            ### step the environment
            t1 = time.time()
            state = env.step(target_qpos)
            t2 = time.time()

            states.append(state)
            actions.append(target_qpos)
            actual_dt_history.append([t0, t1, t2])

            if master_bot_left and master_bot_right:
                state_len = int(len(action) / 2)
                left_action = action[:state_len]
                right_action = action[state_len:]
                master_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
                master_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)

                joint_single_command = JointSingleCommand(name="gripper")

                joint_single_command.cmd = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(left_action[-1])
                master_bot_left.gripper.core.pub_single.publish(joint_single_command)

                joint_single_command.cmd = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(right_action[-1])
                master_bot_right.gripper.core.pub_single.publish(joint_single_command)

        return states, actions, actual_dt_history


def load_policy_and_stats(policy_config, ckpt_path, policy_class='ACT'):
    policy = make_policy(policy_class, policy_config)
    checkpoint = torch.load(ckpt_path)
    loading_status = policy.load_state_dict(checkpoint['model_state_dict'])
    print(loading_status)
    print(f"Loaded: {ckpt_path} val_loss: {checkpoint['val_loss']}")
    stats_path = Path(ckpt_path).parent/Path('dataset_stats.pkl')
    # stats_path = os.path.join(run_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return policy, stats


def eval_bc(config, ckpt_path, num_rollouts, save_episode=False):

    print(f'evaluating {ckpt_path}')

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    policy, stats = load_policy_and_stats(policy_config, ckpt_path, policy_class)

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers, reboot_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True, task=task_name)
        env_max_reward = 1.0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    query_frequency = 20
    num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    checkpoint_info = CheckPointInfo.load(ckpt_path)

    for rollout_id in range(num_rollouts):

        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        initial_state = env.reset()

        print("press any key to continue, q to quit")
        char = getch()
        if char == 'q':
            exit()

        states, actions, timings = [initial_state], [], []
        states, actions, timings = execute_policy_on_env(policy, env, states, actions, timings, max_timesteps,
                                                         state_dim, stats, camera_names, query_frequency=query_frequency, num_queries=num_queries)

        rewards = [s.reward for s in states]
        image_list = [s.observation['images'] for s in states]

        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        print("Space for PASS, any key for FAIL, s for SCRATCH, q for exit")
        char = getch()
        if char == 'q':
            exit()
        if char == ' ':
            print("PASS")
            rewards.append(1.0)
            checkpoint_info.values['trials'].append(1)
        elif char == 's':
            print("SCRATCH")
        else:
            print("FAIL")
            rewards.append(0.0)
            checkpoint_info.values['trials'].append(0)
        checkpoint_info.update()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)

        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    if real_robot:
        from aloha_scripts.robot_utils import move_grippers, reboot_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha
        reboot_grippers(env.puppet_bot_left, env.puppet_bot_right)
        move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
        pass

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate} Average return: {avg_return} of {len(highest_rewards)} trials\n'

    print(f'{checkpoint_info.ckpt_path} : {checkpoint_info.success_rate} {checkpoint_info.successes}/{checkpoint_info.trials_n}')
    # for r in range(env_max_reward+1):
    #     more_or_equal_r = (np.array(highest_rewards) >= r).sum()
    #     more_or_equal_r_rate = more_or_equal_r / num_rollouts
    #     summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + Path(ckpt_path).stem + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, value = data
    image_data, qpos_data, action_data, is_pad, value = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), value.cuda()

    return policy(qpos_data, image_data, action_data, is_pad, value) # TODO remove None


def train_bc(config):

    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']


    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    if config['resume']:
        ckpt_path = find_checkpoint(config['resume'])
        print(f'resuming from {ckpt_path}')
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['val_loss']
        min_inv_learning_error = checkpoint['min_inv_learning_error'] if 'min_inv_learning_error' in checkpoint else np.inf
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        if 'wandb_id' in checkpoint:
            id = checkpoint['wandb_id']
        else:
            id = None
        run = wandb.init(project=f'imitate_{args.task_name}', config=config, id=id, resume='allow')
    else:  # new run
        run = wandb.init(project=f'imitate_{args.task_name}', config=config)
        start_epoch = 0
        min_val_loss = np.inf
        min_inv_learning_error = np.inf

    ckpt_dir = f"{ckpt_dir}/{run.name}"

    task_config = get_task_config(args.task_name)
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    validation_set = task_config['validation_set'] if 'validation_set' in task_config else None
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, args.batch_size, args.batch_size, args.samples_per_epoch,
                                                           cutout_prob=args_dict['cutout_prob'], cutout_patch_size=args_dict['cutout_patch_size'], validation_set=validation_set)

    # save dataset stats
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    def save_policy_last(run, epoch, min_val_loss, min_inv_learning_error):
        ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
        torch.save(
            {
                "epoch": epoch,
                "train_loss": 0,
                "val_loss": min_val_loss,
                "min_inv_learning_error": min_inv_learning_error,
                "model_state_dict": policy.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "wandb_id": run.id
            }, ckpt_path)


    train_history = []
    validation_history = []

    current_epoch = start_epoch

    pbar = tqdm(range(start_epoch, start_epoch + num_epochs))
    for iteration, epoch in enumerate(pbar):
        current_epoch = epoch

        try:
            # training
            policy.train()
            optimizer.zero_grad()
            timing = {'dataload': [], 'forward': [], 'backward': []}

            dataload_ts_start = time.time()
            for batch_idx, data in enumerate(train_dataloader):
                forward_pass_ts_start = time.time()

                forward_dict = forward_pass(data, policy)

                loss = forward_dict['loss']

                backward_ts_start = time.time()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_history.append(detach_dict(forward_dict))
                timing['dataload'] += [forward_pass_ts_start - dataload_ts_start]
                timing['forward'] += [backward_ts_start - forward_pass_ts_start]
                dataload_ts_start = time.time()
                timing['backward'] += [dataload_ts_start - backward_ts_start]

                wandb.log({
                    'train_loss': loss.item(),
                    'train_l1': forward_dict['l1'].item(),
                    'train_Kl': forward_dict['kl'].item(),
                    'train_inv_learning_error': forward_dict['value'].item(),
                })


            # compute train epoch stats
            epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*iteration:(batch_idx+1)*(iteration+1)])
            epoch_train_loss = epoch_summary['loss']
            summary_string = ''
            summary_string += f"epoch: {epoch} " \
                              f"loss: {epoch_train_loss:.5f}, " \
                              f"load_t: {sum(timing['dataload']):.2f}  " \
                              f"fp_t {sum(timing['forward']):.2f} " \
                              f"bp_t {sum(timing['backward']):.2f} "

            # validation
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                    wandb.log({
                        'val_loss': forward_dict['loss'].item(),
                        'val_l1': forward_dict['l1'].item(),
                        'val_Kl': forward_dict['kl'].item(),
                        'val_inv_learning_error': forward_dict['value'].item(),
                    })

                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                epoch_inv_learning_error = epoch_summary['value']

                min_val_loss, min_inv_learning_error = save_best_checkpoints(ckpt_dir, run,
                                                                             epoch_val_loss, min_val_loss,
                                                                             epoch_inv_learning_error, min_inv_learning_error,
                                                                             policy, optimizer, epoch_train_loss, epoch,
                                                                             top_n_checkpoints=3)

                wandb.log({
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_val_loss': epoch_val_loss,
                    'min_val_loss': min_val_loss,
                    'epoch_inv_learning_error': epoch_inv_learning_error,
                    'min_inv_learning_error': min_inv_learning_error
                })

                summary_string += f'Best ckpt, val loss {min_val_loss:.6f} '
                pbar.set_description(summary_string)

                if epoch % 1000 == 0:
                    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
                    torch.save(
                        {
                            "epoch": epoch,
                            "train_loss": epoch_train_loss,
                            "val_loss": min_val_loss,
                            "min_inv_learning_error": min_inv_learning_error,
                            "model_state_dict": policy.state_dict(),
                            "opt_state_dict": optimizer.state_dict(),
                            "wandb_id": run.id
                        }, ckpt_path)
                    plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        except KeyboardInterrupt:
            print(f'Training terminated: seed {seed}, val loss {min_val_loss:.6f}')
            break

    save_policy_last(run, current_epoch, min_val_loss, min_inv_learning_error)
    print(f'Saved checkpoint policy_last at epoch {current_epoch}, best_checkpoint {min_val_loss}')
    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    # print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store', type=str, help='resume_dir', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_path', required=False, default=None, help='path to specific checkpoint for evaluation')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--samples_per_epoch', action='store', type=int, help='number of samples per episode per epoch', default=16)
    parser.add_argument('--cutout_prob', type=float, default=0.0)
    parser.add_argument('--cutout_patch_size', type=int, default=300)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    args = parser.parse_args()

    args_dict = vars(args)
    main(args_dict)
