from abc import abstractmethod, ABC
from typing import Dict

import torch
from omegaconf import DictConfig
from torch import Tensor, tensor
from torch.nn import Module, ModuleDict


class MultiTaskLoss(Module, ABC):
    """A general base class for multitask losses. Takes a list of losses and corresponding keys

    """

    def __init__(self, datasets: DictConfig):
        super().__init__()
        name_to_loss = {name: ds.loss for name, ds in datasets.items()}
        self.losses = ModuleDict(name_to_loss)

    @abstractmethod
    def forward(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]):
        """Aggregate over multiple losses and return a total loss.

        :param y_pred: mapping from dataset name to predictions.
        :param y_true: mapping from dataset name to ground truth.
        :return:
        """


class SumLoss(MultiTaskLoss):
    """Take individual losses for each dataset and simply sum over them.
    """

    def forward(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]):
        total_loss = 0
        for name in y_pred.keys():
            loss = self.losses[name](y_pred[name], y_true[name])
            total_loss += loss

        return total_loss


class MuppetLoss(MultiTaskLoss):
    """Each loss is scaled by log(#target_classes)
    """

    def forward(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]):
        total_loss = 0
        for name in y_pred.keys():
            loss = self.losses[name](y_pred[name], y_true[name]) / torch.log(tensor(y_pred[name].shape[-1]) + 1)
            total_loss += loss

        return total_loss
