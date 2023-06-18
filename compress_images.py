from argparse import ArgumentParser
from pathlib import Path
import h5py
from aloha_scripts.constants import TASK_CONFIGS
from tqdm import tqdm
from PIL import Image
import io
import numpy as np


def compress_image(image, quality=90):
    image = Image.fromarray(image, 'RGB')

    # Create an in-memory byte stream
    output_stream = io.BytesIO()

    # Compress the image to JPEG and save it to the byte stream
    image.save(output_stream, format='JPEG', quality=quality)

    # Get the byte string from the byte stream
    compressed_data = output_stream.getvalue()

    # Close the byte stream
    output_stream.close()

    return compressed_data


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('task_name', help='task_name from aloha.constants')
    parser.add_argument('cache_dir', help='/home/<user>/.cache')
    args = parser.parse_args()

    dataset_dir = Path(TASK_CONFIGS[args.task_name]['dataset_dir'])
    cam_names = TASK_CONFIGS[args.task_name]['camera_names']
    cache_dir = Path(args.cache_dir)
    target_dir = cache_dir / dataset_dir.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

    action                  (14,)         'float64'
    """

    hdf5_files = list(dataset_dir.glob('*.hdf5'))

    pbar = tqdm(hdf5_files)
    for file in pbar:
        pbar.set_description(f'processing: {file}')
        with h5py.File(file, 'r') as root:

            cache_dataset = f'{target_dir}/{file.stem}.hdf5'
            with h5py.File(cache_dataset, 'w') as dest_file:
                observations = dest_file.create_group('observations')
                images = observations.create_group('images')
                root.copy('/action', dest_file['/'])
                root.copy('/observations/qpos', dest_file['/observations'])
                root.copy('/observations/qvel', dest_file['/observations'])
                dest_file.attrs['episode_len'] = TASK_CONFIGS[args.task_name]['episode_len']

                for cam in cam_names:
                    cam_g = images.create_group(cam)
                    image_list = root[f'/observations/images/{cam}']
                    filenames = [f'{i}.jpg' for i, _ in enumerate(image_list)]
                    for fname, image in zip(filenames, image_list):
                        image_compressed = compress_image(image)
                        cam_g.create_dataset(fname, data=np.void(image_compressed))
