import torch
import uuid
from pathlib import Path
from checkpoint import CheckPointInfo
from argparse import ArgumentParser
from tqdm import tqdm

def stamp_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"{checkpoint_path} not found"
    checkpoint_info = CheckPointInfo.load(checkpoint_path)
    if 'guid' in checkpoint_info.values:
        return
    else:
        guid = uuid.uuid4().hex
        checkpoint = torch.load(str(checkpoint_path))
        checkpoint['guid'] = guid
        checkpoint_info.values['guid'] = guid
        checkpoint_info.save()
        torch.save(checkpoint, str(checkpoint_path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint_dir')
    args = parser.parse_args()

    checkpoints_ = list(Path(args.checkpoint_dir).glob("*.ckpt"))

    for checkpoint_path in tqdm(checkpoints_):
        stamp_checkpoint(str(checkpoint_path))