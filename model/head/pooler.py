from torch.nn import Module


class Pooler(Module):
    def __init__(self, in_dim: int, out_dim: int, layer: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = layer


class Average(Pooler):
    def forward(self, inputs):
        x = inputs[self.layer]
        return x.mean(dim=1)
