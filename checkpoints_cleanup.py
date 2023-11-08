from pathlib import Path
from rich.table import Table
from rich.console import Console
from argparse import ArgumentParser
from checkpoint import CheckPointInfo
import time
from tqdm import tqdm
import os


def delete(checkpoint):
    print(f'deleting {str(checkpoint)}')
    os.remove(checkpoint)
    os.remove(f'{str(Path(checkpoint))}.data')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint_dir')
    parser.add_argument('--max_val_loss', type=float, default=None)
    args = parser.parse_args()

    checkpoints = []
    checkpoints += list(Path(args.checkpoint_dir).glob(f'*/*/*.ckpt'))

    pbar = tqdm(checkpoints)
    for checkpoint in pbar:
        pbar.set_description(f'processing {checkpoint}')
        try:
            ckpt_info = CheckPointInfo.load(checkpoint)
        except Exception:
            continue

        if ckpt_info.trials_n == 0:
            age_days = (time.time() - checkpoint.stat().st_mtime) / 60 / 60 / 24
            if age_days > 90:
                delete(checkpoint)
                continue

        if ckpt_info.epoch == 0:
            delete(checkpoint)
            continue

        if args.max_val_loss is not None:
            if ckpt_info.val_loss > args.max_val_loss:
                delete(checkpoint)
                continue