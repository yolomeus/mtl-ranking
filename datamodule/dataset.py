import csv
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from os import path
from random import shuffle, Random
from typing import Tuple

import h5py
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer

from datamodule import DatasetSplit


class PreparedDataset(Dataset, ABC):
    """Dataset to be used within a pytorch lightning datamodule. Exposes interface for preparing data and returning a
    dataset split.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def prepare_data(self):
        """To be called in LightningDatamodule's prepare_data() method.

        :return:
        """

    @abstractmethod
    def _get_split(self, split: DatasetSplit):
        """Return the corresponding split of this dataset.

        :param split: the split to return.
        :return: a pytorch dataset representing the selected split
        """

    def get_split(self, split: DatasetSplit):
        return deepcopy(self._get_split(split))


class MTLDataset(PreparedDataset):
    """Combines multiple datasets into a single Multi-Task dataset.
    """

    def __init__(self, name='MTL', **datasets):
        """
        :param name: name of this dataset.
        :param datasets: named datasets as keyword arguments.
        """
        super().__init__(name)
        self.datasets = list(datasets.values())
        self.name_to_idx = {x: i for i, x in enumerate(list(datasets.keys()))}
        self.num_datasets = len(datasets)

        self._dataset_sizes = None

    def __getitem__(self, indices: Tuple[int, int]):
        """

        :param indices: a tuple of dataset index and sample index for that dataset.
        :return: a dict: {'x': input, 'y': label, 'name': dataset.name}
        """
        dataset_idx, sample_idx = indices
        dataset = self.datasets[dataset_idx]
        x = {'name': dataset.name}
        x.update(dataset[sample_idx])
        return x

    def __len__(self):
        return sum(self.dataset_sizes)

    @property
    def dataset_sizes(self):
        if self._dataset_sizes is None:
            self._dataset_sizes = [len(ds) for ds in self.datasets]
        return self._dataset_sizes

    def prepare_data(self):
        for dataset in self.datasets:
            dataset.prepare_data()

    def _get_split(self, split: DatasetSplit):
        self.datasets = [ds.get_split(split) for ds in self.datasets]
        return self

    def collate(self, dataset_name, batch):
        return self.datasets[self.name_to_idx[dataset_name]].collate(batch)


class DummyDataset(PreparedDataset):
    """Randomly generated dummy dataset.
    """

    def __init__(self, size, num_features, num_classes, name):
        super().__init__(name)
        self._size = size
        self._num_features = num_features
        self._num_classes = num_classes

        self.ds = None

    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def __len__(self):
        return self.ds.__len__()

    def prepare_data(self):
        logging.getLogger(self.__class__.__name__).info('pretending to prepare data...')

    def _get_split(self, split: DatasetSplit):
        self.ds = TensorDataset(torch.rand(self._size, self._num_features),
                                torch.randint(self._num_classes, (self._size,)))

        return self


class TREC2019(PreparedDataset):

    def __init__(self, name, data_file, train_file, val_file, test_file, qrels_file):

        super().__init__(name)
        self.name = name

        self._data_file = to_absolute_path(data_file)

        self._train_file = to_absolute_path(train_file)
        self._val_file = to_absolute_path(val_file)
        self._test_file = to_absolute_path(test_file)

        self._qrels_file = to_absolute_path(qrels_file)

        self.qid_to_pid_to_label = self._read_trec_qrels(self._qrels_file)
        self.current_file = None

        self.split = None

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        with h5py.File(self.current_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]

            og_q_id = torch.tensor(self.get_original_query_id(q_id))
            og_doc_id = torch.tensor(self.get_original_document_id(doc_id))

            if self.split == DatasetSplit.TEST:
                # for the test set we use the qrels file
                label = self.qid_to_pid_to_label[int(og_q_id)].get(int(og_doc_id), 0)
                label = torch.as_tensor([label])
            else:
                label = torch.tensor(fp["labels"][index]).unsqueeze(0).long()

        with h5py.File(self._data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]

        return {'q_id': og_q_id, 'doc_id': og_doc_id, 'x': (query, doc), 'y': label.squeeze()}

    def __len__(self):
        with h5py.File(self.current_file, "r") as fp:
            return len(fp["q_ids"])

    def prepare_data(self):
        pass

    def _get_split(self, split: DatasetSplit):
        self.split = split

        if split == DatasetSplit.TRAIN:
            self.current_file = self._train_file
        elif split == DatasetSplit.VALIDATION:
            self.current_file = self._val_file
        elif split == DatasetSplit.TEST:
            self.current_file = self._test_file
        else:
            raise NotImplementedError()

        return self

    def get_original_query_id(self, q_id: int):
        with h5py.File(self._data_file, "r") as fp:
            return int(fp["orig_q_ids"].asstr()[q_id])

    def get_original_document_id(self, doc_id: int):
        with h5py.File(self._data_file, "r") as fp:
            return int(fp["orig_doc_ids"].asstr()[doc_id])

    @staticmethod
    def _read_trec_qrels(qrels_file):
        qrels = defaultdict(dict)
        with open(qrels_file, 'r') as fp:
            for q_id, _, p_id, label in csv.reader(fp, delimiter=' '):
                q_id, p_id, label = map(int, [q_id, p_id, label])
                qrels[q_id][p_id] = int(label)

        return qrels

    def collate(self, batch):
        queries, docs = zip(*[x['x'] for x in batch])
        tokenized = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt')

        labels = torch.stack([x['y'] for x in batch])
        q_ids, doc_ids = torch.stack([x['q_id'] for x in batch]), torch.stack([x['doc_id'] for x in batch])

        return {'x': tokenized, 'y': labels, 'meta': {'q_ids': q_ids, 'doc_ids': doc_ids}}


class JSONDataset(PreparedDataset):
    def __init__(self, name, raw_file, train_file, val_file, test_file, num_train_samples,
                 num_test_samples):
        super().__init__(name)

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.raw_file = raw_file

        self.train_file = to_absolute_path(train_file)
        self.val_file = to_absolute_path(val_file)
        self.test_file = to_absolute_path(test_file)

        self.data = None

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        x = self.data[index]
        query, doc, label = x['input']['query'], x['input']['passage'], x['target']
        return {'x': (query, doc), 'y': torch.tensor(label)}

    def __len__(self):
        return len(self.data)

    def prepare_data(self):
        raw_dataset = to_absolute_path(self.raw_file)
        to_be_generated = list(map(to_absolute_path,
                                   [self.train_file, self.val_file, self.test_file]))
        if not all(map(path.exists, to_be_generated)):
            with open(raw_dataset, 'r', encoding='utf8') as fp:
                dataset = json.load(fp)

            train_ds, val_ds, test_ds = self._split_ds(dataset)

            with open(self.train_file, 'w', encoding='utf8') as train_fp, \
                    open(self.val_file, 'w', encoding='utf8') as val_fp, \
                    open(self.test_file, 'w', encoding='utf8') as test_fp:
                json.dump(train_ds, train_fp)
                json.dump(val_ds, val_fp)
                json.dump(test_ds, test_fp)

    def _get_split(self, split: DatasetSplit):
        if split == DatasetSplit.TRAIN:
            self.data = self._load_ds(self.train_file)
        elif split == DatasetSplit.VALIDATION:
            self.data = self._load_ds(self.val_file)
        elif split == DatasetSplit.TEST:
            self.data = self._load_ds(self.test_file)
        else:
            raise NotImplementedError

        return self

    def _split_ds(self, dataset):
        shuffle(dataset, Random(5823905).random)
        train_ds, val_ds = dataset[self.num_test_samples:], dataset[:self.num_test_samples]
        train_ds = train_ds[:self.num_train_samples]
        val_ds, test_ds = val_ds[len(val_ds) // 2:], val_ds[:len(val_ds) // 2]

        return train_ds, val_ds, test_ds

    @staticmethod
    def _load_ds(ds_path):
        with open(ds_path, 'r') as fp:
            return json.load(fp)

    def collate(self, batch):
        queries, docs = zip(*[x['x'] for x in batch])
        tokenized = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt')

        labels = torch.stack([x['y'] for x in batch]).unsqueeze(-1)

        return {'x': tokenized, 'y': labels}
