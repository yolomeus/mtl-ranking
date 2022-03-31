"""Training/testing loops specified by pytorch-lightning models. Unlike in standard pytorch-lightning, the loop should
encapsulate the model instead of being bound to it by inheritance. This way, the same model can be trained with
multiple different procedures, without having to duplicate model code by subclassing.
"""
from abc import ABC

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.nn import Module
from torch.optim import Optimizer

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
        self.metrics = MultiTaskMetrics(self.model.output_names,
                                        self.loss,
                                        hparams.metrics.metrics_list,
                                        hparams.metrics.to_probabilities)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        ds_to_inputs, ds_to_y_true, ds_to_meta = batch
        y_pred = self.model(ds_to_inputs)

        loss = self.loss(y_pred, ds_to_y_true)
        total_batch_size = sum([len(x) for x in ds_to_y_true.values()])
        self.log('train/loss', loss.item(), on_step=False, on_epoch=True, batch_size=total_batch_size)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        ds_to_inputs, ds_to_y_true, ds_to_meta = batch
        y_pred = self.model(ds_to_inputs)
        self.metrics.metric_log(self, y_pred, ds_to_y_true, DatasetSplit.VALIDATION)

    def test_step(self, batch, batch_idx):
        ds_to_inputs, ds_to_y_true, ds_to_meta = batch
        y_pred = self.model(ds_to_inputs)
        self.metrics.metric_log(self, y_pred, ds_to_y_true, DatasetSplit.TEST)
