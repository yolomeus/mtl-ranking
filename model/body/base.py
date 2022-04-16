from torch.nn import Module


class MTLBody(Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
