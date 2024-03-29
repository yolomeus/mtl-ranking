from torch.nn import Linear, ReLU, Sequential, Dropout, Sigmoid

from model.head.base import MTLHead, Pooler


class MLP(MTLHead):
    """Simple Multi-Layer Perceptron also known as Feed-Forward Neural Network."""

    def __init__(self, h_dim: int, out_dim: int, dropout: float, pooler: Pooler, sigmoid_out=False):
        """
        :param h_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout rate
        """

        super().__init__(pooler)
        self.classifier = Sequential(Dropout(dropout),
                                     Linear(pooler.out_dim, h_dim),
                                     ReLU(),
                                     Dropout(dropout),
                                     Linear(h_dim, out_dim))
        if sigmoid_out:
            self.classifier.add_module('sig', Sigmoid())

    def _forward(self, inputs):
        x = self.classifier(inputs)
        return x


class DropoutLinear(MTLHead):
    def __init__(self, out_dim: int, dropout: float, pooler: Pooler):
        super().__init__(pooler)
        self.dp = Dropout(dropout)
        self.lin = Linear(pooler.out_dim, out_dim)

    def _forward(self, inputs):
        x = self.dp(inputs)
        return self.lin(x)
