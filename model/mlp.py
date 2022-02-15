from torch.nn import Linear, ReLU, Sequential, Dropout, Module


class MLP(Module):
    """Simple Multi-Layer Perceptron also known as Feed-Forward Neural Network."""

    def __init__(self,
                 in_dim: int,
                 h_dim: int,
                 out_dim: int,
                 dropout: float):
        """

        :param in_dim: input dimension
        :param h_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout rate
        """

        super().__init__()
        self.in_0 = Linear(10, in_dim)
        self.in_1 = Linear(9, in_dim)
        self.in_2 = Linear(7, in_dim)

        self.classifier = Sequential(Dropout(dropout),
                                     Linear(in_dim, h_dim),
                                     ReLU(),
                                     Dropout(dropout),
                                     Linear(h_dim, out_dim))

    def forward(self, inputs):
        if inputs.shape[-1] == 10:
            inputs = self.in_0(inputs)
        elif inputs.shape[-1] == 9:
            inputs = self.in_1(inputs)
        else:
            inputs = self.in_2(inputs)

        x = self.classifier(inputs)
        return x
