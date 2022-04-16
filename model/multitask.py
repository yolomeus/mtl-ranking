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

    def forward(self, name_to_batch: dict):
        """
        :param name_to_batch: a mapping from input module name to the batch of inputs to be passed to the corresponding
        model input.

        :return: a dict mapping from output module names to corresponding predictions.
        """
        preds = {name: self.heads[name](self.body(batch))
                 for name, batch in name_to_batch.items()}
        return preds

    @property
    def output_names(self):
        return self.head.module_names
