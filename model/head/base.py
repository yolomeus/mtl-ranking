from abc import abstractmethod

from torch.nn import Module

from model.head.pooler import Pooler


class MTLHead(Module):
    def __init__(self, pooler: Pooler):
        super().__init__()
        self.pooler = pooler

    def forward(self, inputs, meta: dict):
        return self._forward(self.pooler(inputs, meta))

    @abstractmethod
    def _forward(self, inputs):
        """Implement forward of you module here"""
