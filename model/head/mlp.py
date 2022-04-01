from torch.nn import Linear, ReLU, Sequential, Dropout, Module, Sigmoid

from model.head import MTLHead


class MLP(MTLHead):
    """Simple Multi-Layer Perceptron also known as Feed-Forward Neural Network."""

    def __init__(self, h_dim: int, out_dim: int, dropout: float, pooler: Module, sigmoid_out=False):
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
