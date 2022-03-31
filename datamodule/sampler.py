from random import shuffle
from typing import Iterator

import torch
from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co

from datamodule.dataset import MTLDataset


class RoundRobin(Sampler):
    """Alternately sample separate batches from multiple datasets of an `MTLDataset` until the largest dataset is
    exhausted. Smaller datasets will be padded with uniform sampling to match the largest dataset.
    """

    def __init__(self, batch_size, data_source: MTLDataset, drop_last=False):
        super().__init__(data_source)

        self.num_datasets = data_source.num_datasets
        self.batch_size = batch_size

        self._max_ds_len = max(data_source.dataset_sizes)
        if self._max_ds_len % batch_size != 0:
            if drop_last:
                self._max_ds_len -= self._max_ds_len % batch_size
            else:
                self._max_ds_len += batch_size - (self._max_ds_len % batch_size)

        # generate an index for each dataset with max_ds_len
        # for datasets with length < max_ds_len we sample uniformly
        self.indices = []
        for ds_len in data_source.dataset_sizes:
            ds_idx = torch.randperm(ds_len)

            if ds_len < self._max_ds_len:
                # pad with random samples
                padding = torch.randint(ds_len, (self._max_ds_len - ds_len,))
                ds_idx = torch.cat([ds_idx, padding])

            self.indices.append(ds_idx)

    def __iter__(self) -> Iterator[T_co]:
        batch_idx = 0
        total_idx = 0
        total_len = len(self)
        while total_idx < total_len:
            for ds_idx in range(self.num_datasets):
                # yield enough samples from the current dataset to fill a batch
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size

                for k in range(batch_start, batch_end):
                    sample_idx = self.indices[ds_idx][k]
                    yield ds_idx, int(sample_idx)

                    total_idx += 1

            batch_idx += 1

    def __len__(self):
        return self._max_ds_len * self.num_datasets


class SequentialMultiTask(Sampler):
    """Sequentially iterates over all instances in each dataset of a multitask-dataset.
    """

    def __init__(self, data_source: MTLDataset):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[T_co]:
        for ds_idx in range(self.data_source.num_datasets):
            for sample_idx in range(self.data_source.dataset_sizes[ds_idx]):
                yield ds_idx, sample_idx

    def __len__(self):
        return sum(self.data_source.dataset_sizes)


class RandomProportional(Sampler):
    def __init__(self, data_source: MTLDataset):
        super().__init__(data_source)
        self.data_source = data_source
        self.indexes = []

        for ds_idx in range(self.data_source.num_datasets):
            for sample_idx in range(self.data_source.dataset_sizes[ds_idx]):
                self.indexes.append((ds_idx, sample_idx))

        shuffle(self.indexes)

    def __iter__(self) -> Iterator[T_co]:
        for ds_idx, sample_idx in self.indexes:
            yield ds_idx, sample_idx

    def __len__(self):
        return len(self.indexes)
