from argparse import ArgumentParser
from pathlib import Path
import h5py
from aloha_scripts.constants import TASK_CONFIGS
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import utils
import io
import numpy as np
import cv2


def decode_array_from_dataset(cam_name, timestep):
    image_bytes = root[f'/observations/images/{cam_name}/{timestep}.jpg'][()]
    image = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('directory', help='directory containing datafiles')
    args = parser.parse_args()

    directory = Path(args.directory)
    hdf5_files = list(directory.glob('*.hdf5'))

    for file in hdf5_files:
        with h5py.File(file, 'r') as root:
            print(root.attrs['episode_len'], type(root.attrs['episode_len']))
            for timestep in range(root.attrs['episode_len']):
                images = [decode_array_from_dataset(cam_name, timestep) for cam_name in root['observations/images'].keys()]
                image = np.concatenate(images, axis=1)
                cv2.imshow('compressed', image)
                cv2.waitKey(1)

