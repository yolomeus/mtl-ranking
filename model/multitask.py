from typing import Dict

import torch
from torch.nn import Module, ModuleDict

from model.body.base import MTLBody
from model.head.base import MTLHead


class MultiTaskModel(Module):
    """A model consisting of a body and multiple head modules with the latter being grouped into a `MultiModule`.
    """

    def __init__(self, body: MTLBody, heads: Dict[str, MTLHead]):
        super().__init__()
        self.body = body
        self.heads = ModuleDict(heads)

    def forward(self, name_to_batch: dict, meta: dict):
        """
        :param name_to_batch: a mapping from head name to corresponding batch.
        :param meta: a mapping from head name to any additional data that needs to be passed to.
        :return: a dict mapping from output module names to corresponding predictions.
        """

        preds = {}
        for name in name_to_batch:
            batch = name_to_batch[name]
            if meta[name] and meta[name].get('pairwise'):
                preds[name] = self._pairwise_forward(name, batch, meta)
            else:
                preds[name] = self._dataset_forward(name, batch, meta)

        return preds

    def _dataset_forward(self, name, inputs, meta):
        return self.heads[name](self.body(inputs), meta[name])

    def _pairwise_forward(self, name, inputs, meta):
        preds_pos = self._dataset_forward(name, inputs['pos'], meta)
        preds_neg = self._dataset_forward(name, inputs['neg'], meta)

        return torch.cat([preds_pos, preds_neg], dim=-1)

    @property
    def output_names(self):
        return self.head.module_names
