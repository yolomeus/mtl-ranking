from typing import List

import torch
from torch.nn import Module, MultiheadAttention, Linear, Sequential, Softmax, PReLU


class Pooler(Module):
    """Base Class for Pooler Modules. A pooler takes in
    """

    def __init__(self, in_dim: int, out_dim: int, layers: List[int]):
        """

        :param in_dim: the input hidden dimension expected to be passed to the pooler.
        :param out_dim: the output_dimension of the pooler.
        :param layers: list of layers to pool from.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers


class Average(Pooler):
    """Averages layer outputs over the sequence dimension. If multiple layers are passed, they're averaged over the
    layer dimension first.
    """

    def __init__(self, in_dim: int, out_dim: int, layers: List[int]):
        super().__init__(in_dim, out_dim, layers)
        assert in_dim == out_dim or out_dim is None, 'the average pooler will have the same input dim as output dim'
        self._pool_layers = False
        if len(self.layers) > 1:
            self._pool_layers = True

    def forward(self, inputs):
        """
        :param inputs: a list containing for each layer: a batch of sequences of vectors with size in_dim
        """
        layer_outputs = [inputs[i] for i in self.layers]

        if self._pool_layers:
            x = torch.stack(layer_outputs, dim=-1).mean(-1)
        else:
            x = layer_outputs[0]

        return x.mean(dim=1)


class CLSToken(Pooler):
    """Only get the CLS Token output which is expected to be the first token in the sequence. When pooling multiple
    layers, the average of CLS token outputs will be returned.
    """

    def __init__(self, in_dim: int, out_dim: int, layers: List[int]):
        super().__init__(in_dim, out_dim, layers)
        assert in_dim == out_dim or out_dim is None

    def forward(self, inputs):
        # select CLS token for each layer
        layer_outputs = [inputs[i][:, 0] for i in self.layers]

        # only average if there are multiple cls tokens
        if len(self.layers) > 1:
            x = torch.stack(layer_outputs, -1).mean(-1)
        else:
            x = layer_outputs[0]

        return x


class SelfAttention(Pooler):
    """

    """

    def __init__(self, in_dim: int, out_dim: int, layers: List[int]):
        super().__init__(in_dim, out_dim, layers)
        self.mha = MultiheadAttention(in_dim, 1, dropout=0.1, batch_first=True)
        self.attention_scorer = Sequential(Linear(in_dim, in_dim),
                                           PReLU(),
                                           Linear(in_dim, 1),
                                           Softmax(1))

    def forward(self, inputs):
        x = inputs[self.layers]
        out, _ = self.mha(x, x, x, need_weights=False)
        a = self.attention_scorer(out)
        out = (out * a).sum(1)
        return out
