from typing import Dict

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
        preds = {name: self.heads[name](self.body(batch), meta[name])
                 for name, batch in name_to_batch.items()}
        return preds

    @property
    def output_names(self):
        return self.head.module_names
