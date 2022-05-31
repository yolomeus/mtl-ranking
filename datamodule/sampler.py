import pickle
from os import makedirs
from os.path import exists
from random import shuffle, Random
from typing import Iterator

import h5py
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co
from tqdm import tqdm

from datamodule.dataset import MTLDataset, TREC2019, JSONDataset


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


class LimitedData(Sampler):

    def __init__(self, limit_to: int, data_source: MTLDataset):
        super().__init__(data_source)

        self.cache_file = to_absolute_path('.cache/pair_cache.pkl')
        makedirs(to_absolute_path('.cache'), exist_ok=True)

        self.limit_to = limit_to

        ds_to_idx = data_source.name_to_idx
        trec_ds: TREC2019 = data_source.datasets[ds_to_idx['trec2019']]

        trec_sample_ids = range(len(trec_ds))
        if not exists(self.cache_file):
            trec_qid_pid_pairs = self._get_trec_qid_pid_idx(trec_ds, trec_sample_ids)
            with open(self.cache_file, 'wb') as fp:
                pickle.dump(trec_qid_pid_pairs, fp)
        else:
            with open(self.cache_file, 'rb') as fp:
                trec_qid_pid_pairs = pickle.load(fp)

        self.pairs_to_idx = {ds.name: self._get_json_qid_pid_idx(ds)
                             for ds in data_source.datasets
                             if ds.name != 'trec2019'}

        joint_ds_pairs = set.intersection(*map(lambda x: set(x.keys()), self.pairs_to_idx.values()))
        valid_pairs = joint_ds_pairs.intersection(trec_qid_pid_pairs)

        final_indices = []
        for pair in valid_pairs:
            for ds_name, pair_to_idx in self.pairs_to_idx.items():
                final_indices.append((data_source.name_to_idx[ds_name], pair_to_idx[pair]))

        assert len(final_indices) >= limit_to, f'there are only {len(final_indices)} pairs to sample from.'
        self.final_indices = Random(52317565).sample(final_indices, limit_to)

    def __iter__(self):
        return iter(self.final_indices)

    def __len__(self):
        return len(self.final_indices)

    def _get_trec_qid_pid_idx(self, trec_ds, sample_ids):
        with h5py.File(trec_ds.current_file, 'r') as fp:
            qid_pid_idx = {}
            for idx in tqdm(sample_ids, total=len(sample_ids)):
                _, _, q_id, doc_id = trec_ds.get_ids(fp, idx)
                qid_pid_idx[(int(q_id), int(doc_id))] = idx

        return qid_pid_idx

    def _get_json_qid_pid_idx(self, dataset: JSONDataset):
        qid_pid_to_idx = {(x['info']['qid'], x['info']['pid']): i for i, x in enumerate(dataset.data)}
        return qid_pid_to_idx
