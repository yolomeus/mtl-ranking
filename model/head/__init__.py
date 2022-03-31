from abc import abstractmethod

from torch.nn import Module


class MTLHead(Module):
    def __init__(self, pooler: Module):
        super().__init__()
        self.pooler = pooler

    def forward(self, inputs):
        return self._forward(self.pooler(inputs))

    @abstractmethod
    def _forward(self, inputs):
        """Implement forward of you module here"""
