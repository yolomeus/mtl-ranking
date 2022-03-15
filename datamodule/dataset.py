import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Union, Sized, Tuple

import torch
from torch.utils.data import Dataset, TensorDataset

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

    def __init__(self, datasets: List[Union[PreparedDataset, Sized]], name='MTL'):
        """

        :param datasets:
        """
        super().__init__(name)
        self.datasets = datasets
        self.num_datasets = len(datasets)

        self._dataset_sizes = None

    def __getitem__(self, indices: Tuple[int, int]):
        """

        :param indices: a tuple of dataset index and sample index for that dataset.
        :return: a dict: {'x': input, 'y': label, 'name': dataset.name}
        """
        dataset_idx, sample_idx = indices
        dataset = self.datasets[dataset_idx]
        x, y = dataset[sample_idx]
        return {'x': x, 'y': y, 'name': dataset.name}

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
