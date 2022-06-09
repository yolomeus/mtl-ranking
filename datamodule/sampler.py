import pickle
import random
from collections import defaultdict
from logging import getLogger
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
        shuffle(self.indexes)
        for ds_idx, sample_idx in self.indexes:
            yield ds_idx, sample_idx

    def __len__(self):
        return len(self.indexes)


class LimitedDataFromTrec(Sampler):
    """Samples `limit_to` query - passages pairs from trec, then gets the same pairs from other tasks.
    """

    def __init__(self, data_source: MTLDataset,
                 limit_to: int,
                 non_trec_percentage: float = None):
        """

        :param limit_to: only use this many samples from the trec train dataset.
        :param non_trec_percentage: how many samples to get from the other tasks in percent of limit_to.
        :param data_source: the mtl dataset to sample from.
        :param additional_tasks: the names of tasks to be included for sampling.
        """
        super().__init__(data_source)

        self.LOGGER = getLogger(__name__ + '.' + self.__class__.__name__)

        # we cache the valid trec qid-pid pairs since it takes a while to get them from the dataset
        self.cache_file = to_absolute_path('.cache/pair_cache.pkl')
        makedirs(to_absolute_path('.cache'), exist_ok=True)

        self.limit_to = limit_to
        self.name_pair_idx = self._get_name_pair_idx(data_source)

        # we want to use the same samples regardless of the global random seed,
        # so we can still change it without affecting the samples we're operating on
        sampled_pairs = Random(52317565).sample(set(self.name_pair_idx['trec2019'].keys()), limit_to)
        self.name_to_samples = self._get_name_to_samples(data_source.name_to_idx, sampled_pairs)

        # create pool of non trec indices to sample from
        self.non_trec_samples = []
        for name, samples in self.name_to_samples.items():
            if name != 'trec2019':
                self.non_trec_samples.extend(samples)

        if non_trec_percentage is None:
            self.num_non_trec = len(self.non_trec_samples)
        else:
            self.num_non_trec = min(len(self.non_trec_samples), int(limit_to * non_trec_percentage))

        # log sample sizes
        self.LOGGER.info('samples per task:')
        for name, samples in self.name_to_samples.items():
            self.LOGGER.info(f'{name}: {len(samples)}')
        self.LOGGER.info(f'sampling {self.num_non_trec} / {len(self.non_trec_samples)} samples from additional tasks')

    def __iter__(self):
        final_samples = []
        final_samples.extend(self.name_to_samples['trec2019'])
        non_trec_samples = random.sample(self.non_trec_samples, self.num_non_trec)
        final_samples.extend(non_trec_samples)

        return iter(final_samples)

    def __len__(self):
        return len(self.name_to_samples['trec2019']) + self.num_non_trec

    def _get_name_to_samples(self, name_to_idx, sampled_trec_pairs):
        ds_to_skipped = defaultdict(int)
        name_to_samples = defaultdict(list)

        for ds_name, pair_to_idx in self.name_pair_idx.items():
            ds_idx = name_to_idx[ds_name]
            for pair in sampled_trec_pairs:
                try:
                    sample_idx = pair_to_idx[pair]
                    name_to_samples[ds_name].append((ds_idx, sample_idx))
                except KeyError:
                    ds_to_skipped[ds_name] += 1

        return name_to_samples

    def _get_name_pair_idx(self, datasource):
        datasets = datasource.datasets
        ds_name_to_idx = datasource.name_to_idx

        pairs_to_idx = {ds.name: self._get_json_qid_pid_idx(ds)
                        for ds in datasets
                        if ds.name != 'trec2019'}

        trec_ds: TREC2019 = datasets[ds_name_to_idx['trec2019']]
        pairs_to_idx['trec2019'] = self._get_trec_qid_pid_idx(trec_ds)

        return pairs_to_idx

    @staticmethod
    def _build_trec_qid_pid_idx(trec_ds, sample_ids):
        with h5py.File(trec_ds.current_file, 'r') as fp:
            qid_pid_idx = {}
            for idx in tqdm(sample_ids, total=len(sample_ids)):
                _, _, q_id, doc_id = trec_ds.get_ids(fp, idx)
                qid_pid_idx[(int(q_id), int(doc_id))] = idx

        return qid_pid_idx

    def _get_trec_qid_pid_idx(self, trec_ds):
        trec_size = range(len(trec_ds))
        # build if not exists
        if not exists(self.cache_file):
            trec_qid_pid_idx = self._build_trec_qid_pid_idx(trec_ds, trec_size)
            with open(self.cache_file, 'wb') as fp:
                pickle.dump(trec_qid_pid_idx, fp)
        # read from cache
        else:
            with open(self.cache_file, 'rb') as fp:
                trec_qid_pid_idx = pickle.load(fp)

        return trec_qid_pid_idx

    @staticmethod
    def _get_json_qid_pid_idx(dataset: JSONDataset):
        qid_pid_to_idx = {(int(x['info']['qid']), int(x['info']['pid'])): i
                          for i, x in enumerate(dataset.data)}
        return qid_pid_to_idx
