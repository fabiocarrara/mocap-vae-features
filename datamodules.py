from collections import Counter
import re
from itertools import groupby, chain
from pathlib import Path

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def split_subsequences_from_data_file(data_path):
    subsequence_lines = []
    with open(data_path, 'r') as lines:
        for line in lines:
            if line.lstrip().startswith("#objectKey"):
                if subsequence_lines:
                    yield subsequence_lines
                    subsequence_lines = []
            subsequence_lines.append(line.strip())

        if subsequence_lines:
            yield subsequence_lines


def parse_pose_line(line):
    values = re.split('; |, | ', line)
    values = map(float, values)
    values = np.fromiter(values, dtype=np.float32)
    values = values.reshape(-1, 3)
    return values


def convert_subsequence_to_numpy(subsequence_lines):
    header = subsequence_lines[:2]
    data = subsequence_lines[2:]

    # matches = re.match(r'.*\s(\d+_\d+_\d+_\d+)', header[0])
    # assert matches is not None, f'Error parsing data header: {header[0]}'
    # sample_id = matches.group(0)
    # seq_id, action_id, start_frame, duration = map(int, attributes[1:])
    sample_id = header[0].split(' ')[-1]

    data = map(parse_pose_line, data)
    data = np.stack(list(data))

    expected_frames = int(header[1].split(';')[0])
    assert len(data) == expected_frames, f'Error parsing data: expected {expected_frames} frames, got {len(data)}'

    return sample_id, data


def convert_data_file(data_path):
    subsequences = split_subsequences_from_data_file(data_path)
    subsequences = map(convert_subsequence_to_numpy, tqdm(subsequences))
    subsequences = list(subsequences)

    # drop truncated subsequences
    common_length = Counter(len(s) for _, s in subsequences).most_common(1)[0][0]
    subsequences = filter(lambda x: len(x[1]) == common_length, subsequences)
    sample_ids, subsequences = zip(*subsequences)

    sample_ids = np.array(sample_ids)
    subsequences = np.stack(subsequences)
    return sample_ids, subsequences
    

class MoCapDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=8, seed=7):
        super().__init__()
        self.data_path = Path(data_path)
        self.cache_path = self.data_path.with_suffix('.npz')
        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)
    
    def prepare_data(self):
        if not self.cache_path.exists():
            # convert .data into .npy/pth
            sample_ids, subsequences = convert_data_file(self.data_path)
            np.savez_compressed(self.cache_path, sample_ids=sample_ids, subsequences=subsequences)

    def setup(self, stage=None):
        # load data
        data = np.load(self.cache_path)
        sample_ids, subsequences = data['sample_ids'], data['subsequences']

        # split train/val/test by sequence_id
        samples = pd.DataFrame({'id': sample_ids})
        samples = samples.id.str.rsplit('_', 1, expand=True)
        samples.columns = ['sequence_id', 'subsequence_id']

        grouped = samples.groupby('sequence_id')
        groups = np.array([x for x in grouped.groups])
        self.rng.shuffle(groups)

        n_sequences = len(groups)
        n_train = round(n_sequences * 0.70)
        n_valid = round(n_sequences * 0.15)
        n_test  = n_sequences - n_train - n_valid
        
        train_groups = groups[:n_train]
        valid_groups = groups[n_train:n_train + n_valid]
        test_groups  = groups[-n_test:]

        train_idx = list(chain.from_iterable(grouped.get_group(x).index for x in train_groups))
        valid_idx = list(chain.from_iterable(grouped.get_group(x).index for x in valid_groups))
        test_idx  = list(chain.from_iterable(grouped.get_group(x).index for x in  test_groups))

        self.train_ids   = sample_ids[train_idx]
        self.valid_ids   = sample_ids[valid_idx]
        self.test_ids    = sample_ids[test_idx ]
        self.predict_ids = sample_ids

        self.train_dataset    = TensorDataset(torch.from_numpy(subsequences[train_idx]))
        self.valid_dataset    = TensorDataset(torch.from_numpy(subsequences[valid_idx]))
        self.test_dataset     = TensorDataset(torch.from_numpy(subsequences[ test_idx]))
        self.predict_dataset  = TensorDataset(torch.from_numpy(subsequences))

        print(  "train_dataset len:", len(self.train_dataset))
        print(  "valid_dataset len:", len(self.valid_dataset))
        print(   "test_dataset len:", len(self.test_dataset))
        print("predict_dataset len:", len(self.predict_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass


if __name__ == "__main__":
    data_path = 'data/class130-actions-segment120_shift16-coords_normPOS-fps12.data'
    dm = MoCapDataModule(data_path)
    dm.prepare_data()
    dm.setup()

    train_dataset = dm.train_dataloader()
    breakpoint()
