from itertools import permutations
import math
import torch



def permutation_index(x):
    n = len(x)
    factorials =  torch.tensor([math.factorial(n - i - 1) for i in range(n)], dtype=torch.float) 
    descents = torch.triu((x.unsqueeze(1) > x) * 1.0).sum(dim=1)
    return torch.dot(factorials, descents).to(torch.int)
    
def make_perm(sigma: list[int]):
    return torch.tensor(sigma, dtype=torch.int)

def make_all_perms(N: int):
    return torch.stack([make_perm(p) for p in permutations(range(N))], dim=1)

def _select_perms(idx, perms):
    return torch.index_select(perms, 0, idx)


def _perm_scan(perms, n):
    curr = torch.arange(n, dtype=torch.int)
    out = torch.empty_like(perms)
    for i, p in enumerate(perms):
        curr = curr[p]
        out[i, :] = curr
    return out


perm_scan = torch.vmap(_perm_scan, in_dims=(0, None))
select_perms = torch.vmap(_select_perms, in_dims=(0, None))

def generate_random_perms(Sn, seq_length, batch_dim):
    order, n = Sn.shape
    idx = torch.randint(0, order, (batch_dim, seq_length))
    rand_perms = select_perms(idx, Sn)
    return rand_perms, perm_scan(rand_perms, n)


class SymmetricGroupGenerator:

    def __init__(self, N: int, sequence_length: int, batch_size: int):
        self.N = N
        self.elements = make_all_perms(N)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        #self.device = device
        self.idx_fun = torch.vmap(torch.vmap(permutation_index))
    
    def generate(self):
        perms, labels = generate_random_perms(self.elements, self.sequence_length, self.batch_size)
        return self.idx_fun(perms), self.idx_fun(labels)