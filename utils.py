import cv2
import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader


import IPython

e = IPython.embed


def cutout(mask_size, cutout_prob, cutout_inside=False, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > cutout_prob:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


def read_timestep(episode, ts, camera_names):
    """
    episode: hdf5 file object
    ts: timestep to read
    camera_names: the cam_names to use
    """
    # is_sim = root.attrs['sim']
    episode_len = episode.attrs['episode_len']

    # get observation at start_ts only
    qpos = episode['/observations/qpos'][ts]
    qvel = episode['/observations/qvel'][ts]
    image_dict = dict()
    for cam_name in camera_names:
        image_dict[cam_name] = decompress_image(episode[f'/observations/images/{cam_name}/{ts}.jpg'][()])

    all_cam_images = []
    for cam_name in camera_names:
        all_cam_images.append(image_dict[cam_name])
    all_cam_images = np.stack(all_cam_images, axis=0)

    # get all actions after and including start_ts
    # if is_sim:
    #     action = root['/action'][start_ts:]
    #     action_len = episode_len - start_ts
    # else:
    action = episode['/action'][max(0, ts - 1):]  # hack, to make timesteps more aligned
    action_len = episode_len - max(0, ts - 1)  # hack, to make timesteps more aligned
    # new axis for different cameras

    return qpos, qvel, all_cam_images, action, action_len


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad



class CompressedEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, samples_per_epoch=1,
                 mixup=False, cutout_prob=0., cutout_patch_size=250, discount=0.99):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.samples_per_epoch = samples_per_epoch
        self.mixup = mixup
        self.cutout_prob = cutout_prob
        self.cutout_patch_size = cutout_patch_size
        self.discount = discount
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids) * self.samples_per_epoch

    def __getitem__(self, index):
        index = index % len(self.episode_ids)

        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as episode:
            original_action_shape = episode['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos, qvel, image, action, action_len = read_timestep(episode, start_ts, self.camera_names)

            if self.mixup:
                # lookback = np.random.choice(list(range(8)))
                lookback = 1
                prev_step = max(start_ts - lookback, 0)
                qpos_prev, qvel_prev, image_prev, action_prev, action_len_prev = read_timestep(episode, prev_step, self.camera_names)

                # interpolate
                weight = np.random.uniform()
                image = (image * weight + image_prev * (1. - weight)).astype(np.float32)
                qpos = (qpos * weight + qpos_prev * (1. - weight)).astype(np.float32)
                if action.shape[0] == action_prev.shape[0]:
                    action = (action * weight + action_prev * (1. - weight)).astype(np.float32)
                else:
                    action = (action * weight + action_prev[:action_len] * (1. - weight)).astype(np.float32)

            if self.cutout_prob > 0.:
                cutout_f = cutout(self.cutout_patch_size, self.cutout_prob)
                image = np.stack([cutout_f(im) for im in image])

        # self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1


        # construct observations
        image_data = torch.from_numpy(image)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        monte_carlo_estimate = self.discount ** (episode_len - 1 - start_ts)

        return image_data, qpos_data, action_data, is_pad, monte_carlo_estimate



def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()][:root.attrs['episode_len']]
            qvel = root['/observations/qvel'][()][:root.attrs['episode_len']]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, samples_per_epoch, compressed=True,
              mixup=False, cutout_prob=0., cutout_patch_size=300, validation_set=None):
    print(f'\nData from: {dataset_dir}\n')

    if validation_set is None:
        # obtain train test split by random choice... this might not be great for small datasets
        # as if you can't get a full set of initial states from random choice...
        # then your validation set will be not have full coverage of the initial state
        # and worse, if the training set does not have all initial states, then you will be OOD right away!

        train_ratio = 0.8
        shuffled_indices = np.random.permutation(num_episodes)
        train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
        val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    else:
        # so if dataset is small, hand design your validation set and test set
        val_indices = validation_set
        train_indices = list(set(range(num_episodes)).difference(set(val_indices)))

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    if compressed:
        train_dataset = CompressedEpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, samples_per_epoch=samples_per_epoch,
                                                  mixup=mixup, cutout_prob=cutout_prob, cutout_patch_size=cutout_patch_size)
        val_dataset = CompressedEpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, samples_per_epoch=samples_per_epoch)
    else:
        train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
        val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def decompress_image(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)
