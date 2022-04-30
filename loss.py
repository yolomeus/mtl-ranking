from abc import abstractmethod, ABC
from typing import Dict

import torch
from omegaconf import DictConfig
from torch import Tensor, tensor
from torch.nn import Module, ModuleDict
from torchmetrics.functional import mean_squared_error


class MultiTaskLoss(Module, ABC):
    """A general base class for multitask losses. Takes a list of losses and corresponding keys

    """

    def __init__(self, datasets: DictConfig):
        super().__init__()
        name_to_loss = {ds.name: ds.loss for ds in datasets.values()}
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
        for task_name in y_pred.keys():
            task_loss = self.losses[task_name](y_pred[task_name],
                                               y_true[task_name])

            n_classes = y_pred[task_name].shape[-1]
            if n_classes == 1:
                n_classes += 1
            assert n_classes > 1
            n_classes = tensor(n_classes, device=task_loss.device)

            loss = task_loss / torch.log(n_classes)
            total_loss += loss

        return total_loss


class PairwiseMarginLoss(Module):
    """Pairwise ranking loss expecting a (batch_size, 2) prediction tensor with positive and negative example scores at
    the last dimension and no labels. Returns MSE instead if labels are provided.
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, y_pred, y_true=None):
        """

        :param y_pred: tensor of shape (batch_size, 2) where the last 2 dimensions indicate relevance score for positive
        and negative examples respectively.
        :param y_true: if y_true is provided, MSE between y_pred and y_true will be returned, e.g. for having a loss
        value for validation.
        """
        if y_pred.shape[-1] == 2:
            assert y_true is None, 'ranking loss expects'
            y_pred = y_pred.sigmoid()
            y_pos = y_pred[:, 0]
            y_neg = y_pred[:, 1]
            return torch.mean(torch.clamp(self.margin - y_pos + y_neg, min=0))

        assert y_pred.shape[-1] == 1
        return mean_squared_error(y_pred.sigmoid(), y_true)
