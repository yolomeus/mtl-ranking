from abc import abstractmethod
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from datamodule import DatasetSplit
from datamodule.dataset import MTLDataset
from datamodule.sampler import RoundRobin


class AbstractDefaultDataModule(LightningDataModule):
    """Base class for pytorch-lightning DataModule datasets. Subclass this if you have a standard train, validation,
    test split."""

    def __init__(self, train_conf, test_conf, num_workers, pin_memory):
        super().__init__()

        self._train_conf = train_conf
        self._test_conf = test_conf
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        # setup train and validation datasets
        if stage in (None, 'fit'):
            self._train_ds, self._val_ds = self.train_ds(), self.val_ds()
        # setup test dataset
        if stage in (None, 'test'):
            self._test_ds = self.test_ds()

    @abstractmethod
    def train_ds(self):
        """Build the train pytorch dataset.

        :return: the train pytorch dataset.
        """

    @abstractmethod
    def val_ds(self):
        """Build the validation pytorch dataset.

        :return: the validation pytorch dataset.
        """

    @abstractmethod
    def test_ds(self):
        """Build the test pytorch dataset.

        :return: the test pytorch dataset.
        """

    def train_dataloader(self):
        train_dl = DataLoader(self._train_ds,
                              self._train_conf.batch_size,
                              shuffle=True,
                              num_workers=self._num_workers,
                              pin_memory=self._pin_memory,
                              collate_fn=self.build_collate_fn(DatasetSplit.TRAIN),
                              persistent_workers=self._num_workers > 0)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self._val_ds,
                            self._test_conf.batch_size,
                            num_workers=self._num_workers,
                            pin_memory=self._pin_memory,
                            collate_fn=self.build_collate_fn(DatasetSplit.VALIDATION),
                            persistent_workers=self._num_workers > 0)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self._test_ds,
                             self._test_conf.batch_size,
                             num_workers=self._num_workers,
                             pin_memory=self._pin_memory,
                             collate_fn=self.build_collate_fn(DatasetSplit.TEST),
                             persistent_workers=self._num_workers > 0)
        return test_dl

    # noinspection PyMethodMayBeStatic
    def build_collate_fn(self, split: DatasetSplit = None):
        """Override to define a custom collate function. Build a function for collating multiple data instances into
        a batch. Defaults to returning `None` since it's the default for DataLoader's collate_fn argument.

        While different collate functions might be needed depending on the dataset split, in most cases the same
        function can be returned for all data splits.

        :param split: The split that the collate function is used on to build batches. Can be ignored when train and
        test data share the same structure.
        :return: a single argument function that takes a list of tuples/instances and returns a batch as tensor or a
        tuple of multiple batch tensors.
        """

        return None


class MTLDataModule(AbstractDefaultDataModule):
    def __init__(self, train_conf, test_conf, num_workers, pin_memory):
        super().__init__(train_conf, test_conf, num_workers, pin_memory)

    def train_ds(self):
        return MTLDataset([TensorDataset(torch.rand(12808, 10), torch.randint(10, (12808,))),
                           TensorDataset(torch.rand(64003, 9), torch.randint(10, (64003,))),
                           TensorDataset(torch.rand(32004, 7), torch.randint(10, (32004,)))])

    def val_ds(self):
        return TensorDataset(torch.rand(128, 10), torch.randint(10, (128,)))

    def test_ds(self):
        return TensorDataset(torch.rand(128, 10), torch.randint(10, (128,)))

    def train_dataloader(self):
        train_dl = DataLoader(self._train_ds,
                              self._train_conf.batch_size,
                              num_workers=self._num_workers,
                              pin_memory=self._pin_memory,
                              collate_fn=self.build_collate_fn(DatasetSplit.TRAIN),
                              persistent_workers=self._num_workers > 0,
                              sampler=RoundRobin(self._train_conf.batch_size, self._train_ds))
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self._val_ds,
                            self._test_conf.batch_size,
                            num_workers=self._num_workers,
                            pin_memory=self._pin_memory,
                            collate_fn=self.build_collate_fn(DatasetSplit.VALIDATION),
                            persistent_workers=self._num_workers > 0)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self._test_ds,
                             self._test_conf.batch_size,
                             num_workers=self._num_workers,
                             pin_memory=self._pin_memory,
                             collate_fn=self.build_collate_fn(DatasetSplit.TEST),
                             persistent_workers=self._num_workers > 0)
        return test_dl
