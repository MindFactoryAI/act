from matplotlib import pyplot as plt
from argparse import ArgumentParser

from utils import CompressedEpisodicDataset, get_norm_stats
from aloha_scripts.constants import TASK_CONFIGS
import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('task_name')
    args = parser.parse_args()

    if args.task_name in TASK_CONFIGS:
        task = TASK_CONFIGS[args.task_name]
    else:
        print(f'{args.task_name} not found in aloha_scripts.constants')
        exit()

    # get the dataset
    norm_stats = get_norm_stats(task['dataset_dir'], task['num_episodes'])
    indices = list(range(task['num_episodes']))
    ds = CompressedEpisodicDataset(indices, task['dataset_dir'], task['camera_names'], norm_stats, mixup=False, cutout=True)

    image, qpos, action, is_pad, value = ds[0]
    image = torch.cat(image.unbind(0), dim=2)

    # setup the grid
    fig, [image_ax, action_ax, pad_ax] = plt.subplots(3, 1)
    image_ax.imshow(image.permute(1, 2, 0))
    action = action * ~is_pad.unsqueeze(-1)
    for joint in action.T:
        action_ax.plot(joint)
    pad_ax.plot(is_pad.T)
    plt.show()

