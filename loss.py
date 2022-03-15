from abc import abstractmethod, ABC
from typing import List, Dict

from torch import Tensor
from torch.nn import Module, ModuleDict


class MultiTaskLoss(Module, ABC):
    """A general base class for multitask losses. Takes a list of losses and corresponding keys

    """

    def __init__(self, dataset_names: List[str], dataset_losses: List[Module]):
        """

        :param dataset_names: name (str key) of each dataset to compute loss over.
        :param dataset_losses: the loss module for each individual dataset.
        """
        super().__init__()
        self.dataset_names = dataset_names
        self.dataset_losses = dataset_losses

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

    def __init__(self, dataset_names, dataset_losses):
        super().__init__(dataset_names, dataset_losses)

        self.losses = ModuleDict({name: loss
                                  for name, loss in zip(self.dataset_names, self.dataset_losses)})

    def forward(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]):
        total_loss = 0
        for name in y_pred.keys():
            loss = self.losses[name](y_pred[name], y_true[name])
            total_loss += loss

        return total_loss
