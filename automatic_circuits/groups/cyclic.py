import torch
from torch.distributions import Categorical, Dirichlet


def generate_cum_addition(seq_len: int, n: int, batch_size: int):
    summands = torch.empty((batch_size, seq_len), dtype=torch.int32)
    for i in range(batch_size): 
        probs = Dirichlet(torch.ones(n,)).sample()
        dist = Categorical(probs)
        summands[i, :] = dist.sample((seq_len,))
    sums = summands.cumsum(dim=-1) % n
    return summands, sums



class CyclicGroupGenerator:

    def __init__(self, seq_len: int, N: int, batch_size: int, device):
        self.seq_len = seq_len
        self.N = N
        self.batch_size = batch_size
        self.device = device
        self.gen_fn = torch.compile(generate_cum_addition)

    def generate(self):
        summands, sums = self.gen_fn(self.seq_len, self.N, self.batch_size)
        return summands.to(self.device), sums.to(self.device)
    

class CyclicGroupGeneratorScratchpad:

    def __init__(self, seq_len: int, N: int, batch_size: int):
        self.seq_len = seq_len
        self.N = N
        self.sep = torch.full((batch_size, 1), N * 2)
        self.batch_size = batch_size
        self.gen_fn = torch.compile(generate_cum_addition)

    def generate(self):
        summands, sums = self.gen_fn(self.seq_len, self.N, self.batch_size)
        sums += self.N
        seq = torch.stack([summands, sums], dim=2).reshape(self.batch_size, -1)
        return torch.stack([self.sep, seq], dim=1)
