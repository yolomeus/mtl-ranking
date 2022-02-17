import torch
from torch.utils.data import TensorDataset, DataLoader

from datamodule import DatasetSplit
from datamodule.dataset import MTLDataset
from datamodule.default_datamodule import AbstractDefaultDataModule
from datamodule.sampler import RoundRobin


class MTLDataModule(AbstractDefaultDataModule):
    def __init__(self, train_conf, test_conf, num_workers, pin_memory):
        super().__init__(train_conf, test_conf, num_workers, pin_memory)

    def prepare_data(self) -> None:
        self._train_ds.prepare_data()
        self._val_ds.prepare_data()
        self._test_ds.prepare_data()

    def train_ds(self) -> MTLDataset:
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
