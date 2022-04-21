import csv
import json
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy, copy
from os import path
from random import shuffle, Random
from typing import Tuple, Dict

import h5py
import torch
from hydra.utils import to_absolute_path
from torch.nn import ModuleDict, Module
from torch.utils.data import Dataset

from datamodule import DatasetSplit
from datamodule.preprocessor import Preprocessor


class PreparedDataset(Dataset, ABC):
    """Dataset to be used within a pytorch lightning datamodule. Exposes interface for preparing data and returning a
    dataset split.
    """

    def __init__(self,
                 name: str,
                 metrics: Dict[str, ModuleDict] = None,
                 to_probabilities: Module = None,
                 loss: Module = None,
                 preprocessor: Preprocessor = None):
        """

        :param name: the name of the dataset.
        :param metrics: mapping from split name (train, val, test) to a ModuleDict that maps from
        :param to_probabilities: the module used for converting predictions to a probability distribution to fit this
        dataset's labels.
        :param loss: the loss module to be used for this dataset.
        :param preprocessor: the preprocessor to be applied to the input data.
        """
        if metrics is not None:
            # only keep names of metrics
            self.metrics = {split_name: [m for m in split] for split_name, split in metrics.items()}
        if to_probabilities is not None:
            self.to_probabilities = str(to_probabilities)

        self.loss = loss

        self.name = name
        self.preprocessor = preprocessor

    @abstractmethod
    def prepare_data(self):
        """To be called in LightningDatamodule's prepare_data() method. Used for downloading, pre-processing or similar.
        Do not assign state from this method.
        """

    @abstractmethod
    def _get_split(self, split: DatasetSplit):
        """Return the corresponding split of this dataset.

        :param split: the split to return.
        :return: a pytorch dataset representing the selected split.
        """

    def get_split(self, split: DatasetSplit):
        return deepcopy(self._get_split(split))


class MTLDataset(PreparedDataset):
    """Combines multiple datasets into a single Multi-Task dataset.
    """

    def __init__(self, datasets: Dict[str, PreparedDataset], name='MTL'):
        """
        :param datasets: mapping from dataset names to datasets.
        :param name: name of this dataset.
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


class TREC2019(PreparedDataset):

    def __init__(self,
                 data_file: str,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 qrels_file_test: str,
                 qrels_file_val: str,
                 name: str,
                 metrics: Dict[str, ModuleDict],
                 to_probabilities: Module,
                 loss: Module,
                 preprocessor: Preprocessor):

        super().__init__(name, metrics, to_probabilities, loss, preprocessor)
        self.name = name

        self._data_file = to_absolute_path(data_file)

        self._train_file = to_absolute_path(train_file)
        self._val_file = to_absolute_path(val_file)
        self._test_file = to_absolute_path(test_file)

        self._qrels_file_test = to_absolute_path(qrels_file_test)
        self._qrels_file_val = to_absolute_path(qrels_file_val)

        self.qrels_test = self._read_trec_qrels(self._qrels_file_test, DatasetSplit.TEST)
        self.qrels_val = self._read_trec_qrels(self._qrels_file_val, DatasetSplit.VALIDATION)

        self.current_file = None
        self.split = None

    def __getitem__(self, index):
        with h5py.File(self.current_file, 'r') as fp:
            q_id, doc_id, og_q_id, og_doc_id = self.get_ids(fp, index)
            label, label_rank = self.get_label(fp, index, og_q_id, og_doc_id)

        query, doc = self.get_query_and_doc(q_id, doc_id)

        x = {'q_id': og_q_id, 'doc_id': og_doc_id, 'x': (query, doc), 'y': label.squeeze()}
        if label_rank is not None:
            x['y_rank'] = label_rank

        return x

    def __len__(self):
        with h5py.File(self.current_file, "r") as fp:
            return len(fp['q_ids'])

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

    def get_ids(self, fp, index):
        q_id = fp['q_ids'][index]
        doc_id = fp['doc_ids'][index]

        og_q_id = torch.tensor(self.get_original_query_id(q_id))
        og_doc_id = torch.tensor(self.get_original_document_id(doc_id))

        return q_id, doc_id, og_q_id, og_doc_id

    def get_query_and_doc(self, q_id, doc_id):
        with h5py.File(self._data_file, 'r') as fp:
            query = fp['queries'].asstr()[q_id]
            doc = fp['docs'].asstr()[doc_id]

        return query, doc

    def get_label(self, fp, index, og_q_id, og_doc_id):
        label = None
        label_rank = None
        if self.split == DatasetSplit.TEST:
            label_rank = self.get_qrel(og_q_id, og_doc_id, self.qrels_test)
        elif self.split == DatasetSplit.VALIDATION:
            label_rank = self.get_qrel(og_q_id, og_doc_id, self.qrels_val)
        else:
            label = torch.tensor([fp['labels'][index]], dtype=torch.long)

        if label_rank is not None:
            label = torch.tensor(int(label_rank > 0))

        return label, label_rank

    def get_original_query_id(self, q_id: int):
        with h5py.File(self._data_file, 'r') as fp:
            return int(fp['orig_q_ids'].asstr()[q_id])

    def get_original_document_id(self, doc_id: int):
        with h5py.File(self._data_file, 'r') as fp:
            return int(fp['orig_doc_ids'].asstr()[doc_id])

    def get_qrel(self, og_q_id, og_doc_id, qrels):
        label = qrels[int(og_q_id)].get(int(og_doc_id), 0)
        return torch.tensor(label)

    @staticmethod
    def _read_trec_qrels(qrels_file, split: DatasetSplit):
        # qrels for val is .tsv while test is .txt
        if split == DatasetSplit.TEST:
            delimiter = ' '
        else:
            delimiter = '\t'

        qrels = defaultdict(dict)
        with open(qrels_file, 'r', encoding='utf8') as fp:
            for q_id, _, p_id, label in csv.reader(fp, delimiter=delimiter):
                q_id, p_id, label = map(int, [q_id, p_id, label])
                qrels[q_id][p_id] = int(label)

        return qrels

    def collate(self, batch):
        tokenized = self.preprocessor(batch)

        labels = torch.stack([x['y'] for x in batch])
        q_ids = torch.stack([x['q_id'] for x in batch])

        x = {'x': tokenized, 'y': labels, 'meta': {'indexes': q_ids}}

        if batch[0].get('y_rank', None) is not None:
            y_rank = torch.stack([x['y_rank'] for x in batch])
            x['meta']['y_rank'] = y_rank

        return x


class TREC2019Pairwise(TREC2019):

    def __getitem__(self, index):
        if self.split != DatasetSplit.TRAIN:
            return super().__getitem__(index)

        with h5py.File(self.current_file, "r") as fp:
            q_id = fp['q_ids'][index]
            pos_doc_id = fp['pos_doc_ids'][index]
            neg_doc_id = fp['neg_doc_ids'][index]

        with h5py.File(self._data_file, "r") as fp:
            query = fp['queries'].asstr()[q_id]
            pos_doc = fp['docs'].asstr()[pos_doc_id]
            neg_doc = fp['docs'].asstr()[neg_doc_id]

        x = {'query': query, 'pos_doc': pos_doc, 'neg_doc': neg_doc}
        return x

    def collate(self, batch):
        if self.split != DatasetSplit.TRAIN:
            return super().collate(batch)

        pos_batch = []
        neg_batch = []
        for x in batch:
            x_pos = {'x': (x['query'], x['pos_doc'])}
            x_neg = {'x': (x['query'], x['neg_doc'])}
            pos_batch.append(x_pos)
            neg_batch.append(x_neg)

        pos_tokenized = self.preprocessor(pos_batch)
        neg_tokenized = self.preprocessor(neg_batch)

        return {'x': {'pos': pos_tokenized, 'neg': neg_tokenized}, 'y': None, 'meta': {'pairwise': True}}


class JSONDataset(PreparedDataset, ABC):
    """In-Memory dataset that reads all instances from a json file.
    """

    def __init__(self,
                 raw_file: str,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 num_train_samples: int,
                 num_test_samples: int,
                 name: str,
                 metrics: Dict[str, ModuleDict],
                 to_probabilities: Module,
                 loss: Module,
                 preprocessor: Preprocessor):
        """

        :param raw_file: the full json file of samples
        :param train_file: the json file containing the train instances. Will be generated if it doesn't exist yet.
        :param val_file: the json file containing the validation instances. Will be generated if it doesn't exist yet.
        :param test_file: the json file containing the test instances. Will be generated if it doesn't exist yet.
        :param num_train_samples: the number of samples in the train split.
        :param num_test_samples: the combined number of samples in the validation and test split. Will be split 50/50.
        """
        super().__init__(name, metrics, to_probabilities, loss, preprocessor)

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.raw_file = raw_file

        self.train_file = to_absolute_path(train_file)
        self.val_file = to_absolute_path(val_file)
        self.test_file = to_absolute_path(test_file)

        self.data = None

    def __len__(self):
        return len(self.data)

    def _split_dataset(self, dataset):
        assert len(dataset) >= self.num_test_samples + self.num_train_samples
        shuffle(dataset, Random(5823905).random)
        train_ds, val_ds = dataset[self.num_test_samples:], dataset[:self.num_test_samples]
        train_ds = train_ds[:self.num_train_samples]
        val_ds, test_ds = val_ds[len(val_ds) // 2:], val_ds[:len(val_ds) // 2]

        return train_ds, val_ds, test_ds

    def _dump_splits(self, train_ds, val_ds, test_ds):
        with open(self.train_file, 'w', encoding='utf8') as train_fp, \
                open(self.val_file, 'w', encoding='utf8') as val_fp, \
                open(self.test_file, 'w', encoding='utf8') as test_fp:
            json.dump(train_ds, train_fp)
            json.dump(val_ds, val_fp)
            json.dump(test_ds, test_fp)

    def _get_split(self, split: DatasetSplit):
        if split == DatasetSplit.TRAIN:
            self.data = self._load_dataset(self.train_file)
        elif split == DatasetSplit.VALIDATION:
            self.data = self._load_dataset(self.val_file)
        elif split == DatasetSplit.TEST:
            self.data = self._load_dataset(self.test_file)
        else:
            raise NotImplementedError

        return self

    @staticmethod
    def _load_dataset(ds_path):
        with open(ds_path, 'r') as fp:
            return json.load(fp)


class RegressionJSONDataset(JSONDataset):
    """JSONDataset with regression targets. Takes care of normalizing the targets and provides option to bucketize, in
    order to cast to a classification task.
    """

    def __init__(self,
                 raw_file: str,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 num_train_samples: int,
                 num_test_samples: int,
                 name: str,
                 metrics: Dict[str, ModuleDict],
                 to_probabilities: Module,
                 loss: Module,
                 preprocessor: Preprocessor,
                 num_buckets: int = 1):
        """
        :param num_buckets: the number of buckets
        """
        super().__init__(raw_file, train_file, val_file, test_file, num_train_samples, num_test_samples, name, metrics,
                         to_probabilities, loss, preprocessor)
        self._num_buckets = num_buckets

    def __getitem__(self, index):
        x = self.data[index]
        query, doc, label = x['input']['query'], x['input']['passage'], x['target']
        label = torch.tensor(label)

        if self._num_buckets > 1:
            label = self.bucketize(label)

        return {'x': (query, doc), 'y': label}

    def prepare_data(self):
        raw_dataset = to_absolute_path(self.raw_file)
        to_be_generated = list(map(to_absolute_path,
                                   [self.train_file, self.val_file, self.test_file]))
        if not all(map(path.exists, to_be_generated)):
            with open(raw_dataset, 'r', encoding='utf8') as fp:
                dataset = json.load(fp)

            for x in dataset:
                x['target'] = x['targets'][0]['label']
                del x['targets']

            train_ds, val_ds, test_ds = self._split_dataset(dataset)
            train_ds, val_ds, test_ds = self.normalize_targets(train_ds, val_ds, test_ds)

            self._dump_splits(train_ds, val_ds, test_ds)

    @staticmethod
    def normalize_targets(train_ds, val_ds, test_ds):
        train_max = max([x['target'] for x in train_ds])
        train_min = min([x['target'] for x in train_ds])
        for x in train_ds:
            x['target'] = (x['target'] - train_min) / (train_max - train_min)

        for ds in [val_ds, test_ds]:
            for x in ds:
                x['target'] = max(train_min, min(x['target'], train_max))
                x['target'] = (x['target'] - train_min) / (train_max - train_min)

        return train_ds, val_ds, test_ds

    def bucketize(self, label):
        boundaries = torch.linspace(0, 1, self._num_buckets)[1:]
        return torch.bucketize(label, boundaries)

    def collate(self, batch):
        tokenized = self.preprocessor(batch)
        labels = torch.stack([x['y'] for x in batch])
        if self._num_buckets == 1:
            labels = labels.unsqueeze(-1)

        return {'x': tokenized, 'y': labels}


class ClassificationJSONDataset(JSONDataset):
    def __init__(self,
                 raw_file: str,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 label_file: str,
                 num_train_samples: int,
                 num_test_samples: int,
                 num_classes: int,
                 name: str,
                 metrics: Dict[str, ModuleDict],
                 to_probabilities: Module,
                 loss: Module,
                 preprocessor: Preprocessor):
        """
        :param num_classes:
        """
        super().__init__(raw_file, train_file, val_file, test_file, num_train_samples, num_test_samples, name, metrics,
                         to_probabilities, loss, preprocessor)
        self._num_classes = num_classes
        self._label_file = to_absolute_path(label_file)

        self._label_to_id = None

    def __getitem__(self, index):
        if self._label_to_id is None:
            self._load_label_to_id()

        x = copy(self.data[index])
        x['y'] = torch.tensor(self._label_to_id[x['y']], dtype=torch.long)
        return x

    def prepare_data(self):
        raw_dataset = to_absolute_path(self.raw_file)
        to_be_generated = list(map(to_absolute_path,
                                   [self.train_file, self.val_file, self.test_file]))
        if not all(map(path.exists, to_be_generated)):
            with open(raw_dataset, 'r', encoding='utf8') as fp:
                dataset = json.load(fp)

            labels = set()
            new_dataset = []
            for x in dataset:
                for target in x['targets']:
                    labels.add(target['label'])

                    x_new = {'x': (x['input']['query'], x['input']['passage']),
                             'y': target['label'],
                             'span1': target['span1']}
                    if 'span2' in target:
                        x_new['span2'] = target['span2']

                    new_dataset.append(x_new)

            train_ds, val_ds, test_ds = self._split_dataset(new_dataset)
            self._dump_splits(train_ds, val_ds, test_ds)

            with open(self._label_file, 'wb') as fp:
                pickle.dump({label: i for i, label in enumerate(labels)}, fp)

    def _load_label_to_id(self):
        with open(self._label_file, 'rb') as fp:
            self._label_to_id = pickle.load(fp)

    def collate(self, batch):
        tokenized, new_spans = self.preprocessor(batch)
        labels = torch.stack([x['y'] for x in batch])
        return {'x': tokenized, 'y': labels, 'meta': {'spans': torch.tensor(new_spans, dtype=torch.long)}}
