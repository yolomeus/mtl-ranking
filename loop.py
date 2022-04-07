"""Training/testing loops specified by pytorch-lightning models. Unlike in standard pytorch-lightning, the loop should
encapsulate the model instead of being bound to it by inheritance. This way, the same model can be trained with
multiple different procedures, without having to duplicate model code by subclassing.
"""
from abc import ABC
from typing import Dict

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from transformers import get_constant_schedule_with_warmup

from datamodule import DatasetSplit
from logger.utils import MultiTaskMetrics
from model.multitask import MultiTaskModel


class AbstractBaseLoop(LightningModule, ABC):
    """Abstract base class for implementing a training loop for a pytorch model.
    """

    def __init__(self, hparams: DictConfig):
        """
        :param hparams: contains all hyperparameters to be logged before training.
        """
        super().__init__()
        self.save_hyperparameters(hparams)


# noinspection PyAbstractClass
class MultiTaskLoop(AbstractBaseLoop):
    """Default wrapper for training/testing a pytorch module using pytorch-lightning. Assumes a standard classification
    task with instance-label pairs (x, y) and a loss function that has the signature loss(y_pred, y_true).
    """

    def __init__(self, hparams: DictConfig, model: MultiTaskModel, optimizer: Optimizer, loss: Module):
        super().__init__(hparams)

        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        dataset_cfgs = hparams.datamodule.dataset.datasets
        self.metrics = MultiTaskMetrics(dataset_cfgs)

    def configure_optimizers(self):
        lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, self.hparams.training.warmup_steps)
        return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}

    def training_step(self, batch, batch_idx):
        inputs, y_true, meta = batch
        y_pred = self.model(inputs)
        loss = self.loss(y_pred, y_true)
        return {'loss': loss, 'y_true': y_true, 'y_pred': self._detach(y_pred), 'meta': meta}

    def training_step_end(self, outputs):
        with torch.no_grad():
            self.metrics(self, outputs['y_pred'], outputs['y_true'], outputs['meta'], DatasetSplit.TRAIN)

    def validation_step(self, batch, batch_idx):
        inputs, y_true, meta = batch
        y_pred = self.model(inputs)
        return {'y_true': y_true, 'y_pred': self._detach(y_pred), 'meta': meta}

    def validation_step_end(self, outputs):
        self.metrics(self, outputs['y_pred'], outputs['y_true'], outputs['meta'], DatasetSplit.VALIDATION)

    def test_step(self, batch, batch_idx):
        inputs, y_true, meta = batch
        y_pred = self.model(inputs)
        return {'y_true': y_true, 'y_pred': self._detach(y_pred), 'meta': meta}

    def test_step_end(self, outputs):
        self.metrics(self, outputs['y_pred'], outputs['y_true'], outputs['meta'], DatasetSplit.TEST)

    @staticmethod
    def _detach(preds: Dict[str, Tensor]):
        return {k: v.detach() for k, v in preds.items()}
