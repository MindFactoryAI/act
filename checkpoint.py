import json
from pathlib import Path

import torch


class TestResult:
    def __init__(self, episode, result):
        self._result = result





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


def get_resume_checkpoint(ckpt_dir):
    checkpoints = list(Path(ckpt_dir).glob('*.ckpt'))
    for ckpt in checkpoints:
        if ckpt.name == 'policy_last.ckpt':
            return str(ckpt)
    else:
        print(f"No checkpoints found in {ckpt_dir}")
        exit(1)


def list_checkpoints(checkpoint_dir, prefix=None):
    if prefix is None:
        return sorted(list(Path(checkpoint_dir).glob('*.ckpt')))
    else:
        return sorted(list(Path(checkpoint_dir).glob(f'{prefix}*.ckpt')))
