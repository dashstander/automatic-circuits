import torch
from torch.distributions import Categorical


def generate_cum_addition(seq_len: int, n: int, batch_size: int):
    summands = torch.empty((batch_size, seq_len), dtype=torch.int32)
    dist = Categorical(torch.ones((n,)))
    summands = dist.sample((batch_size, seq_len))
    #for i in range(batch_size): 
        #probs = Dirichlet(torch.ones(n,)).sample()
    #summands[i, :] = dist.sample((seq_len,))
    sums = summands.cumsum(dim=-1) % n
    return summands, sums


class CyclicGroupGenerator:

    def __init__(self, seq_len: int, N: int, batch_size: int):
        self.seq_len = seq_len
        self.N = N
        self.batch_size = batch_size
        self.gen_fn = generate_cum_addition
        self.sep = torch.full((batch_size, 1), N)

    def _add_sep(self, tensors):
        return tuple([
          torch.concatenate([self.sep, tensor], dim=1) for tensor in tensors  
        ])

    def generate(self):
        return self._add_sep(
            self.gen_fn(self.seq_len, self.N, self.batch_size)
        )
    

class CyclicGroupGeneratorScratchpad:

    def __init__(self, seq_len: int, N: int, batch_size: int):
        self.seq_len = seq_len
        self.N = N
        self.sep = torch.full((batch_size, 1), N * 2)
        self.batch_size = batch_size
        self.gen_fn = torch.compile(generate_cum_addition)

    @property
    def order(self):
        return self.N

    def generate(self):
        summands, sums = self.gen_fn(self.seq_len, self.N, self.batch_size)
        sums += self.N
        seq = torch.stack([summands, sums], dim=2).reshape(self.batch_size, -1)
        if torch.randn(()) > 0:
            return torch.concatenate([self.sep, self.sep, seq], dim=1)
        else:
            return torch.concatenate([self.sep, seq, self.sep], dim=1)
