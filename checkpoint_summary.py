from pathlib import Path
from rich.table import Table
from rich.console import Console
from argparse import ArgumentParser
from checkpoint import CheckPointInfo
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint_dir')
    parser.add_argument('task_name', default='*')
    args = parser.parse_args()

    checkpoints = []
    checkpoints += list(Path(args.checkpoint_dir).glob(f'{args.task_name}/*/*.ckpt'))
    checkpoints += list(Path(args.checkpoint_dir).glob('*.ckpt'))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("Checkpoint", style="cyan", no_wrap=True)
    table.add_column("Val Loss", style="cyan", no_wrap=True)
    table.add_column("Pass", style="green")
    table.add_column("Trials", style="red")
    table.add_column("Rate")

    pbar = tqdm(checkpoints)
    for checkpoint in pbar:
        pbar.set_description(f'processing {checkpoint}')
        ckpt_info = CheckPointInfo.load(checkpoint)
        if ckpt_info.trials_n > 0:
            table.add_row(
                f'{checkpoint.parent.parent.name}',
                f'{checkpoint.parent.name}',
                f'{checkpoint.name}',
                f'{ckpt_info.val_loss:.5f}',
                f'{ckpt_info.successes}',
                f'{ckpt_info.trials_n}',
                f'{ckpt_info.success_rate}',
            )

    console = Console(width=180)
    console.print(table)