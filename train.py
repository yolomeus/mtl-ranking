import os

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from callbacks import TestPredictionWriter


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed, workers=True)

    model = instantiate(cfg.model)
    training_loop = instantiate(cfg.loop,
                                # hparams for saving
                                cfg,
                                model=model,
                                # pass model params to optimizer constructor
                                optimizer={"params": model.parameters()})

    train_cfg = cfg.training
    ckpt_path = os.path.join(os.getcwd(), 'checkpoints/')
    filename = 'epoch-{epoch:03d}-' + train_cfg.monitor.replace('/', '_') + '-{' + train_cfg.monitor + ':.4f}'
    model_checkpoint = ModelCheckpoint(save_top_k=train_cfg.save_ckpts,
                                       monitor=train_cfg.monitor,
                                       mode=train_cfg.mode,
                                       verbose=True,
                                       filename=filename,
                                       auto_insert_metric_name=False,
                                       dirpath=ckpt_path)

    early_stopping = EarlyStopping(monitor=train_cfg.monitor,
                                   patience=train_cfg.patience,
                                   mode=train_cfg.mode,
                                   verbose=True)

    if cfg.logger is not None:
        logger = instantiate(cfg.logger)
        if cfg.log_gradients:
            logger.experiment.watch(training_loop.model)
    else:
        # setting to True will use the default logger
        logger = True

    preds_dir = os.path.join(os.getcwd(), 'predictions/')
    trainer = Trainer(max_epochs=train_cfg.epochs,
                      gpus=cfg.gpus,
                      logger=logger,
                      callbacks=[model_checkpoint, early_stopping, LearningRateMonitor(),
                                 TestPredictionWriter(preds_dir)],
                      accumulate_grad_batches=train_cfg.accumulate_batches,
                      precision=train_cfg.precision)

    # do not instantiate datamodule recursively,
    # we need to pass the dataset object to the sampler constructors
    datamodule = instantiate(cfg.datamodule,
                             _recursive_=False)

    trainer.fit(training_loop, datamodule=datamodule)

    # only look at this in the very end ;)
    trainer.test(ckpt_path='best', datamodule=datamodule)

    if issubclass(logger.__class__, WandbLogger):
        wandb.finish()


if __name__ == '__main__':
    train()
