import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Sized

import torch
from torch.utils.data import Dataset, TensorDataset


class PreparedDataset(Dataset, ABC):
    @abstractmethod
    def prepare_data(self):
        """

        :return:
        """

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """

        :param stage:
        :return:
        """


class MTLDataset(PreparedDataset):
    """Combines multiple datasets into a single Multi-Task dataset.
    """

    def __init__(self, datasets: List[Union[PreparedDataset, Sized]]):
        """

        :param datasets:
        """
        self.datasets = datasets
        self.dataset_sizes = [len(ds) for ds in datasets]
        self.num_datasets = len(datasets)

    def __getitem__(self, indices):
        """

        :param indices:
        :return:
        """
        dataset_idx, sample_idx = indices
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.dataset_sizes

    def prepare_data(self):
        for dataset in self.datasets:
            dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        for dataset in self.datasets:
            dataset.setup(stage)


class DummyDataset(PreparedDataset):
    def __init__(self, size, num_features, num_classes):
        self._size = size
        self._num_features = num_features
        self._num_classes = num_classes

        self.ds = None

    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def __len__(self):
        return self.ds.__len__()

    def prepare_data(self):
        self.ds = TensorDataset(torch.rand(self._size, self._num_features),
                                torch.randint(self._num_classes, (self._size,)))

    def setup(self, stage: Optional[str] = None):
        logging.getLogger(self.__class__.__name__).info('pretending to set something up...')
