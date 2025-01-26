from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import pandas as pd
from io import BytesIO
from PIL import Image

class AgiAlpha2LerobotDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'cam_top_depth': tfds.features.Image(
                            shape=(480, 640, 1),
                            dtype=np.float32, # TODO: 아래에서 tiff -> float 32
                            encoding_format='png', # virtualkss: 'jpeg' or 'png'. Format to serialize np.ndarray images on disk. If None, encode images as PNG. If image is loaded from {bmg,gif,jpeg,png} file, this parameter is ignored, and file original encoding is used.
                            doc='cam top depth.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(20, ),
                            dtype=np.float32,
                            doc='state',
                        ),
                        'frame_index': tfds.features.Scalar(
                            dtype=np.int64,
                            doc='frame index',
                        ),
                        'timestamp': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='timestamp',
                        ),
                        # TODO: check the others
                    }),
                    'action': tfds.features.Tensor(
                        shape=(22,),
                        dtype=np.float32,
                        doc='action',
                    ),

                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_index': tfds.features.Scalar(
                        doc='episode index',
                        dtype=np.int64
                    ),
                    'task_index': tfds.features.Scalar(
                        doc='task index',
                        dtype=np.int64
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.parquet'),
            #'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def convert_tiff_bytes_to_float32(byte_data):
            """
            바이트 데이터를 np.float32로 변환
            """
            # BytesIO를 사용해 메모리에서 TIFF 데이터 읽기
            with Image.open(BytesIO(byte_data)) as img:
                # TIFF 데이터를 NumPy 배열로 변환
                img_array = np.array(img, dtype=np.float32)
            return img_array

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = pd.read_parquet(episode_path) # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, row in data.iterrows(): # for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([''])[0].numpy()

                episode.append({
                    'observation': {
                        'cam_top_depth': convert_tiff_bytes_to_float32(row['observation.images.cam_top_depth']['bytes'])[:, :, np.newaxis],
                        'state': row['observation.state'],
                        'frame_index': row['frame_index'],
                        'timestamp': row['timestamp'],
                    },
                    'action': row['action'],
                    'discount': 1.0,
                    'reward': 0, #float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': '', #['language_instruction'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path, 
                    'episode_index': data.iloc[0, 3], 
                    'task_index': data.iloc[0, 6]
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

