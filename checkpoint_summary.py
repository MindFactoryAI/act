from pathlib import Path
from rich.table import Table
from rich.console import Console
from argparse import ArgumentParser
from imitate_episodes import CheckPointInfo

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint_dir')
    args = parser.parse_args()

    checkpoints = list(Path(args.checkpoint_dir).glob('*/*/*.ckpt'))
    checkpoints += list(Path(args.checkpoint_dir).glob('*.ckpt'))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("Checkpoint", style="cyan", no_wrap=True)
    table.add_column("Val Loss", style="cyan", no_wrap=True)
    table.add_column("Pass", style="green")
    table.add_column("Trials", style="red")
    table.add_column("Rate")

    for checkpoint in checkpoints:
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

    console = Console(width=120)
    console.print(table)