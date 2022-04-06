from functools import partial
from typing import Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datamodule import DatasetSplit
from datamodule.dataset import PreparedDataset


class MTLDataModule(LightningDataModule):
    def __init__(self,
                 dataset: PreparedDataset,
                 train_batch_size: int,
                 test_batch_size: int,
                 num_workers: int,
                 pin_memory: bool,
                 train_sampler: DictConfig,
                 val_sampler: DictConfig,
                 test_sampler: DictConfig):
        super().__init__()

        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size

        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._dataset = instantiate(dataset)

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

        self._train_sampler_cfg = train_sampler
        self._val_sampler_cfg = val_sampler
        self._test_sampler_cfg = test_sampler

        self._train_sampler = None
        self._val_sampler = None
        self._test_sampler = None

    def prepare_data(self) -> None:
        self._dataset.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_ds = self._dataset.get_split(DatasetSplit.TRAIN)
        self._val_ds = self._dataset.get_split(DatasetSplit.VALIDATION)
        self._test_ds = self._dataset.get_split(DatasetSplit.TEST)

        self._train_sampler = instantiate(self._train_sampler_cfg, data_source=self._train_ds)
        self._val_sampler = instantiate(self._val_sampler_cfg, data_source=self._val_ds)
        self._test_sampler = instantiate(self._test_sampler_cfg, data_source=self._test_ds)

    def train_dataloader(self):
        train_dl = DataLoader(self._train_ds,
                              self._train_batch_size,
                              num_workers=self._num_workers,
                              pin_memory=self._pin_memory,
                              collate_fn=self.build_collate_fn(DatasetSplit.TRAIN),
                              persistent_workers=self._num_workers > 0,
                              sampler=self._train_sampler)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self._val_ds,
                            self._test_batch_size,
                            num_workers=self._num_workers,
                            pin_memory=self._pin_memory,
                            collate_fn=self.build_collate_fn(DatasetSplit.VALIDATION),
                            persistent_workers=self._num_workers > 0,
                            sampler=self._val_sampler)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self._test_ds,
                             self._test_batch_size,
                             num_workers=self._num_workers,
                             pin_memory=self._pin_memory,
                             collate_fn=self.build_collate_fn(DatasetSplit.TEST),
                             persistent_workers=self._num_workers > 0,
                             sampler=self._test_sampler)
        return test_dl

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

        return partial(collate, dataset=self._dataset)


def collate(batch, dataset):
    dataset_to_inputs = {}
    dataset_to_labels = {}
    dataset_to_meta = {}

    ds_names = set([x['name'] for x in batch])
    for name in ds_names:
        batch_items = [x for x in batch if x['name'] is name]
        dataset_batch_dict = dataset.collate(name, batch_items)

        dataset_to_inputs[name] = dataset_batch_dict['x']
        dataset_to_labels[name] = dataset_batch_dict['y']
        dataset_to_meta[name] = dataset_batch_dict.get('meta', None)

    return dataset_to_inputs, dataset_to_labels, dataset_to_meta
