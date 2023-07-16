from argparse import ArgumentParser
import pickle
from utils import load_data
from aloha_scripts.constants import TASK_CONFIGS
import os

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('task_name')
    parser.add_argument('run_dir')
    args = parser.parse_args()

    if not args.task_name in TASK_CONFIGS:
        print(f'{args.task_name} not found')
        exit()

    task = TASK_CONFIGS[args.task_name]

    _, _, stats, _ = load_data(task['dataset_dir'], task['num_episodes'], task['camera_names'], 2, 2, 2)
    print(stats)

    stats_path = os.path.join(args.run_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        cached_stats = pickle.load(f)

    for key in stats:
        assert (stats[key] == cached_stats[key]).all()


