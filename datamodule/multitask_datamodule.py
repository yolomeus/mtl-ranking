from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datamodule import DatasetSplit
from datamodule.dataset import DummyDataset, MTLDataset
from datamodule.sampler import RoundRobin


class MTLDataModule(LightningDataModule):
    def __init__(self, train_conf, test_conf, num_workers, pin_memory):
        super().__init__()

        self._train_conf = train_conf
        self._test_conf = test_conf
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._dataset = MTLDataset([DummyDataset(12808, 10, 10, 'dummy_00'),
                                    DummyDataset(64003, 9, 10, 'dummy_01'),
                                    DummyDataset(32004, 7, 10, 'dummy_02')])

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def prepare_data(self) -> None:
        self._dataset.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_ds = self._dataset.get_split(DatasetSplit.TRAIN)
        self._val_ds = self._dataset.get_split(DatasetSplit.VALIDATION)
        self._test_ds = self._dataset.get_split(DatasetSplit.TEST)

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

    @staticmethod
    def build_collate_fn(split: DatasetSplit = None):
        """Override to define a custom collate function. Build a function for collating multiple data instances into
        a batch. Defaults to returning `None` since it's the default for DataLoader's collate_fn argument.

        While different collate functions might be needed depending on the _dataset split, in most cases the same
        function can be returned for all data splits.

        :param split: The split that the collate function is used on to build batches. Can be ignored when train and
        test data share the same structure.
        :return: a single argument function that takes a list of tuples/instances and returns a batch as tensor or a
        tuple of multiple batch tensors.
        """

        def collate(batch):
            print('todo')
            return batch

        return collate
