import csv
import os
from os import path
from typing import Optional, Any

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class TestPredictionWriter(Callback):
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.has_header = {}

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        for ds_name in outputs['meta']:
            with open(path.join(self.output_dir, f'{ds_name}.csv'), 'a', encoding='utf8', newline='') as fp:
                if not self.has_header.get(ds_name):
                    if ds_name.startswith('trec2019'):
                        fp.write('q_id,doc_id,y_pred,y_true\n')
                    else:
                        fp.write('y_pred,y_true\n')
                    self.has_header[ds_name] = True

                writer = csv.writer(fp)
                preds = outputs['y_pred'][ds_name]

                if ds_name.startswith('trec2019'):
                    meta = outputs['meta'][ds_name]
                    batch_meta = batch[2][ds_name]
                    if len(preds.shape) == 2 and preds.shape[-1] == 2:
                        preds = preds[:, 1]
                    to_write = [batch_meta['indexes'], batch_meta['doc_ids'], preds, meta['y_rank']]
                else:
                    to_write = [preds.argmax(-1), outputs['y_true'][ds_name]]

                # write predictions
                for tensors in zip(*to_write):
                    tensors = list(map(lambda x: x.item(), tensors))
                    writer.writerow(tensors)
