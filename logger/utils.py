"""Training loop related utilities.
"""
from copy import deepcopy
from typing import List, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict, Softmax, Sigmoid, Identity

from datamodule import DatasetSplit


class Metrics(Module):
    """Stores and manages metrics during training/testing for log.
    """

    def __init__(self, loss: Module, metrics_configs: List[DictConfig], to_probabilities: str):
        """

        :param loss: the loss module for computing the loss metric.
        :param metrics_configs: dict configs for each metric to instantiate.
        :param to_probabilities: either 'sigmoid', 'softmax' or 'identity' (None), used to convert raw model predictions into
        probabilities before passing them into a metric function.
        """
        super().__init__()

        per_split_metrics = [[] if metrics_configs is None else [instantiate(metric) for metric in metrics_configs]
                             for _ in range(3)]
        self.train_metrics, self.val_metrics, self.test_metrics = [ModuleList(metrics) for metrics in per_split_metrics]
        self.loss = loss

        if to_probabilities == 'sigmoid':
            self._to_probabilities = Sigmoid()
        elif to_probabilities == 'softmax':
            self._to_probabilities = Softmax(dim=-1)

        elif to_probabilities in [None, 'identity']:
            self._to_probabilities = Identity()

    def forward(self, loop, y_pred, y_true, split: DatasetSplit):
        y_prob = self._to_probabilities(y_pred)

        if split == DatasetSplit.TRAIN:
            metrics = self.train_metrics
        elif split == DatasetSplit.TEST:
            metrics = self.test_metrics
        else:
            metrics = self.val_metrics

        for metric in metrics:
            metric(y_prob, y_true)
            loop.log(f'{split.value}/' + self.classname(metric),
                     metric,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(y_true))

        loss = self.loss(y_pred, y_true)
        loop.log(f'{split.value}/loss', loss, on_step=False, on_epoch=True, batch_size=len(y_true))

        if split == DatasetSplit.TRAIN:
            return loss

    def metric_log(self, loop, y_pred, y_true, split: DatasetSplit):
        return self.forward(loop, y_pred, y_true, split)

    @staticmethod
    def classname(obj, lower=True):
        """Get the classname of an object.

        :param obj: any python object.
        :param lower: return the name in lowercase.
        :return: the classname as string.
        """
        name = obj.__class__.__name__
        return name.lower() if lower else name


class MultiTaskMetrics(Metrics):
    def __init__(self, model_output_names: List[str],
                 loss: Module,
                 metrics_configs: List[DictConfig],
                 to_probabilities: str):
        """

        :param model_output_names:
        :param loss:
        :param metrics_configs:
        :param to_probabilities:
        """
        super().__init__(loss, metrics_configs, to_probabilities)

        self.model_output_names = model_output_names

        self.output_to_train = ModuleDict({name: deepcopy(self.train_metrics) for name in model_output_names})
        self.output_to_val = ModuleDict({name: deepcopy(self.val_metrics) for name in model_output_names})
        self.output_to_test = ModuleDict({name: deepcopy(self.test_metrics) for name in model_output_names})

    def forward(self, loop: LightningModule, y_pred: Dict[str, Tensor], y_true: Tensor, split: DatasetSplit):

        for name in y_pred.keys():
            y_prob = self._to_probabilities(y_pred[name])

            if split == DatasetSplit.TRAIN:
                metrics = self.output_to_train[name]
            elif split == DatasetSplit.TEST:
                metrics = self.output_to_val[name]
            else:
                metrics = self.output_to_test[name]

            for metric in metrics:
                metric(y_prob, y_true)
                loop.log(f'{split.value}/' + self.classname(metric),
                         metric,
                         on_step=False,
                         on_epoch=True,
                         batch_size=len(y_true))

        loss = self.loss(y_pred, y_true)
        loop.log(f'{split.value}/loss', loss, on_step=False, on_epoch=True, batch_size=len(y_true))

        if split == DatasetSplit.TRAIN:
            return loss
