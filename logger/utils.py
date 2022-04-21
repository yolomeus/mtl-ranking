"""Training loop related utilities.
"""
import inspect
from collections import defaultdict

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module, ModuleDict

from datamodule import DatasetSplit


class MultiTaskMetrics(Module):
    def __init__(self, dataset_cfgs: DictConfig):
        super().__init__()

        self.metrics = self._build_metric_module(dataset_cfgs)
        self.to_probabilites = ModuleDict({ds_name: instantiate(ds.to_probabilities)
                                           for ds_name, ds in dataset_cfgs.items()})

    def forward(self, loop, y_pred, y_true, meta, split: DatasetSplit):
        total_batch_size = sum([len(x) for x in y_pred.values()])

        for dataset_name in y_pred.keys():
            y_prob = self.to_probabilites[dataset_name](y_pred[dataset_name]).detach()

            metrics = self._select_metrics(dataset_name, split)
            for name, metric in metrics.items():
                meta_args = self._get_meta_args(metric, dataset_name, meta)
                metric.update(y_prob, y_true[dataset_name], **meta_args)

                loop.log(f'{split.value}/{name}/{dataset_name}',
                         metric,
                         on_step=False,
                         on_epoch=True,
                         batch_size=len(y_pred[dataset_name]))

        loss = loop.loss(y_pred, y_true)
        loop.log(f'{split.value}/loss', loss.detach(), on_step=True, on_epoch=True, batch_size=total_batch_size)

    def _select_metrics(self, dataset_name, split):
        if split == DatasetSplit.TRAIN:
            return self.metrics[dataset_name]['train_metrics']
        elif split == DatasetSplit.TEST:
            return self.metrics[dataset_name]['test_metrics']

        return self.metrics[dataset_name]['val_metrics']

    @staticmethod
    def _build_metric_module(dataset_cfgs):
        ds_to_metrics = defaultdict(dict)
        for ds_name, ds_cfg in dataset_cfgs.items():
            for split, metric_cfgs in ds_cfg.metrics.items():
                ds_to_metrics[ds_name][split] = ModuleDict(
                    {m_name: instantiate(m) for m_name, m in metric_cfgs.items()})

            ds_to_metrics[ds_name] = ModuleDict(ds_to_metrics[ds_name])

        return ModuleDict(ds_to_metrics)

    @staticmethod
    def _get_meta_args(metric, dataset_name, meta):
        meta_args = {}
        if meta[dataset_name] is not None:
            valid_args = inspect.signature(metric.update).parameters.keys()
            meta_args = {k: v for k, v in meta[dataset_name].items() if k in valid_args}

        return meta_args
