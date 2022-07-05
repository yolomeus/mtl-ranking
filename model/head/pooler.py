from abc import abstractmethod, ABC
from typing import List

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU, Softmax


class Pooler(Module, ABC):
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

    def forward(self, inputs, meta=None):
        """

        :param inputs:
        :param meta:
        :return:
        """
        pooled_layers = [self.sequence_aggregate(inputs[i], meta)
                         for i in self.layers]

        if len(pooled_layers) > 1:
            return self.layer_aggregate(pooled_layers, meta)

        return pooled_layers[0]

    @abstractmethod
    def sequence_aggregate(self, single_layer_out, meta):
        """

        :param single_layer_out: list of layers after performing pooling on each.
        :param meta:
        :return: aggregation of all layers
        """

    @abstractmethod
    def layer_aggregate(self, pooled_layers, meta):
        """

        :param pooled_layers: list of layers after performing pooling on each.
        :param meta:
        :return: aggregation of all layers
        """


class SpanPooler(Pooler, ABC):
    """Pooler that uses spans to pool along the sequence dimension.
    """

    def sequence_aggregate(self, single_layer_out, meta=None):
        spans = meta['spans']

        # pair of span batches stacked at dim=0
        if len(spans.shape) == 3:
            s1, s2 = self.collect_span_pairs(spans, single_layer_out)

            s1 = torch.stack([self.span_aggregate(t) for t in s1])
            s2 = torch.stack([self.span_aggregate(t) for t in s2])

            pooled_layer = torch.cat([s1, s2], dim=-1)
            return pooled_layer

        # single batch of spans
        elif len(spans.shape) == 2:
            x = self.collect_spans(single_layer_out, meta['spans'])

            pooled_layer = torch.stack([self.span_aggregate(t) for t in x])
            return pooled_layer

        raise NotImplementedError

    @staticmethod
    def collect_spans(x, spans):
        """Given a sequence of vectors and a list of integer spans, select the vectors within each span.

        :param x:
        :param spans:
        :return: a list of vector sequences, with each sequence corresponding to a span from spans.
        """
        a, b = spans[:, 0], spans[:, 1]
        return [x[i, a[i]:b[i]] for i in torch.arange(0, spans.shape[0])]

    def collect_span_pairs(self, spans, x):
        spans1, spans2 = spans[0], spans[1]
        x1 = self.collect_spans(x, spans1)
        x2 = self.collect_spans(x, spans2)
        return x1, x2

    @abstractmethod
    def span_aggregate(self, spans) -> Tensor:
        """Aggregation function to be applied to each span.

        :return:
        """


class SpanAverage(SpanPooler):
    def span_aggregate(self, span) -> Tensor:
        return torch.mean(span, dim=0)

    def layer_aggregate(self, pooled_layers, meta):
        return torch.mean(torch.stack(pooled_layers), dim=0)


class AttentionAverage(SpanAverage):
    """Average along sequence dim, use attention weights across layers.
    """

    def __init__(self, att_h_dim: int, in_dim: int, out_dim: int, layers: List[int]):
        super().__init__(in_dim, out_dim, layers)
        assert len(layers) > 1
        self.attention = Sequential(Linear(out_dim, att_h_dim),
                                    ReLU(),
                                    Linear(att_h_dim, 1),
                                    Softmax(dim=1))

    def sequence_aggregate(self, single_layer_out, meta=None):
        if meta is not None and 'spans' in meta:
            return super().sequence_aggregate(single_layer_out, meta)
        return torch.mean(single_layer_out, dim=1)

    def layer_aggregate(self, pooled_layers, meta):
        # attention
        pooled_layers = torch.stack(pooled_layers, dim=1)
        att_scores = self.attention(pooled_layers)

        x = (att_scores * pooled_layers).sum(1)
        return x


class Average(Pooler):
    """Averages layer outputs over the sequence dimension and layer dimension.
    """

    def sequence_aggregate(self, single_layer_out, meta):
        return torch.mean(single_layer_out, dim=1)

    def layer_aggregate(self, pooled_layers, meta):
        return torch.mean(torch.stack(pooled_layers), dim=0)


class CLSToken(Pooler):
    """Only get the CLS Token output which is expected to be the first token in the sequence. When pooling multiple
    layers, the average of CLS token outputs will be returned.
    """

    def __init__(self, in_dim: int, out_dim: int, layers: List[int]):
        super().__init__(in_dim, out_dim, layers)
        assert in_dim == out_dim or out_dim is None

    def sequence_aggregate(self, single_layer_out, meta):
        return single_layer_out[:, 0]

    def layer_aggregate(self, pooled_layers, meta):
        return torch.mean(torch.stack(pooled_layers), dim=0)
