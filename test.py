import os
from os import path
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from callbacks import TestPredictionWriter


@hydra.main(config_path='conf', config_name='config')
def test(calling_cfg: DictConfig):
    """Load a model from an old config and a checkpoint file, then test it on the current datamodule.
    """

    test_cfg = calling_cfg.testing
    ckpt_path = to_absolute_path(test_cfg.ckpt_path)
    cfg_file_path = path.join(Path(ckpt_path).parent.parent, '.hydra/config.yaml')

    # config belonging to the old checkpoint
    loaded_cfg = OmegaConf.load(cfg_file_path)
    # replace current model config with loaded one
    calling_cfg['model'] = loaded_cfg['model']
    calling_cfg['loop']['loss'] = loaded_cfg['loop']['loss']

    # instantiate training components
    model = instantiate(calling_cfg.model)
    loop = instantiate(calling_cfg.loop,
                       # hparams for saving
                       calling_cfg,
                       model=model,
                       # pass model params to optimizer constructor
                       optimizer={"params": model.parameters()})

    datamodule = instantiate(calling_cfg.datamodule,
                             _recursive_=False)

    if calling_cfg.logger is not None:
        if test_cfg.run_id is not None:
            logger = instantiate(calling_cfg.logger, id=test_cfg.run_id, resume="allow")
        else:
            logger = instantiate(calling_cfg.logger)

        if calling_cfg.log_gradients:
            logger.experiment.watch(loop.model)
    else:
        # setting to True will use the default logger
        logger = True

    preds_dir = path.join(os.getcwd(), 'predictions/')
    trainer = Trainer(max_epochs=1,
                      gpus=calling_cfg.gpus,
                      logger=logger,
                      callbacks=[TestPredictionWriter(preds_dir)],
                      precision=test_cfg.precision)

    trainer.test(loop, ckpt_path=ckpt_path, datamodule=datamodule)


if __name__ == '__main__':
    test()
