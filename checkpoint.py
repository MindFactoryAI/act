import json
import os
from pathlib import Path
from uuid import uuid4

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

    @property
    def guid(self):
        return self.values['guid']

    @property
    def trials(self):
        if 'trials' not in self.values:
            self.values['trials'] = []
        return self.values['trials']

    @staticmethod
    def load(ckpt_path, human=False):
        checkpoint = Path(ckpt_path)
        if not checkpoint.exists() and not human:
            raise Exception(f'File {checkpoint} not found')

        sidecar = Path(f'{str(Path(ckpt_path))}.data')

        if not sidecar.exists():
            values = {
                'trials': [],
            }
            if human:
                values['guid'] = 'human'
            with open(sidecar, 'w') as file:
                json.dump(values, file)
        else:
            with open(sidecar, 'r') as file:
                try:
                    values = json.load(file)
                except:
                    print(f'failed to load {file}')
                    raise Exception('failed to load checkpoint')

        return CheckPointInfo(ckpt_path, values)

    def save(self):
        with open(self.sidecar, 'w') as file:
            json.dump(self.values, file)

    @property
    def trials_n(self):
        return len(self.trials)

    @property
    def successes(self):
        return sum(self.trials)

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


def save_checkpoint(checkpoint_path, policy, optimizer, sidecar_dict):
    guid = uuid4().hex
    sidecar_dict['guid'] = guid
    checkpoint_path = Path(checkpoint_path)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "guid": guid
    }, str(checkpoint_path))
    sidecar = CheckPointInfo(checkpoint_path, sidecar_dict)
    sidecar.save()


def save_best_checkpoints(checkpoint_dir, run,
                          val_loss, min_val_loss,
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

        sidecar_dict = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": min_val_loss,
            "wandb_id": run.id
        }

        checkpoint_path = f'{checkpoint_dir}/policy_min_val_loss{val_loss:.5f}.ckpt'
        save_checkpoint(checkpoint_path, policy, optimizer, sidecar_dict)

    return min_val_loss
