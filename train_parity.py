import torch
import numpy as np
from tqdm import trange
from transformer_lens import HookedTransformerConfig, HookedTransformer
from torch.utils.data import IterableDataset, DataLoader
import wandb


def get_seq_lengths(total_length, min_length, max_length, rng):
    too_many = rng.integers(min_length, max_length, min_length * total_length)
    sequence_lengths = too_many[np.cumsum(too_many) <= total_length]
    diff = total_length - np.sum(sequence_lengths)
    sequence_lengths = sequence_lengths.tolist()
    if diff >= min_length and diff <= max_length:
        sequence_lengths += [diff]
    return sequence_lengths
    

def generate_cum_parity(total_seq_len: int, rng):
    seq_len = (total_seq_len - 2) // 2
    assert seq_len > 0, total_seq_len
    equals = np.array([2])
    sep = 3
    x = rng.integers(0, 2, seq_len)
    running_x = np.cumsum(x) % 2
    seq = np.concatenate([x, equals, running_x], axis=0)
    seq = np.pad(seq, (1, total_seq_len - len(seq) - 1), mode='constant', constant_values=(sep, sep))
    return seq


def generate_packed_parity(total_seq_length, min_seq_length=6, max_seq_length=30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sep = 3
    sequence_lengths = get_seq_lengths(total_seq_length, min_seq_length, max_seq_length, rng)
    assert sum(sequence_lengths) <= total_seq_length
    parities = [
            generate_cum_parity(seq_len, stream) for seq_len, stream in zip(sequence_lengths, rng.spawn(len(sequence_lengths)))
    ]
    parities = np.concatenate(parities)
    diff = total_seq_length - len(parities)
    return np.pad(parities, (0, diff), mode='constant', constant_values=(sep, sep))


def generate_fixed_parity(total_seq_length, seq_length=30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sep = 3
    sequence_lengths = [seq_length] * (total_seq_length // seq_length)
    assert sum(sequence_lengths) <= total_seq_length
    parities = [
            generate_cum_parity(seq_len, stream) for seq_len, stream in zip(sequence_lengths, rng.spawn(len(sequence_lengths)))
    ]
    parities = np.concatenate(parities)
    diff = total_seq_length - len(parities)
    return np.pad(parities, (0, diff), mode='constant', constant_values=(sep, sep))



class CumulativeParityDataset(IterableDataset):

    def __init__(self, total_sequence_length: int, min_sequence_length: int, max_sequence_length: int, batch_size: int, rng_seed: int = 0):
        super().__init__()
        self.total_sequence_length = total_sequence_length
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(rng_seed)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            all_rngs = [stream for stream in self.rng.spawn(worker_info.num_workers)]
            rng = all_rngs[worker_info.id]
        else:
            rng = self.rng
        while True:
            parities = [
                generate_packed_parity(
                    self.total_sequence_length,
                    self.min_sequence_length,
                    self.max_sequence_length,
                    stream
                ) for stream in rng.spawn(self.batch_size)
            ]
            yield torch.asarray(np.stack(parities, axis=0))



class CumulativeParityFixed(IterableDataset):
    def __init__(self, total_sequence_length: int, sequence_length: int, batch_size: int, rng_seed: int = 0):
        super().__init__()
        self.total_sequence_length = total_sequence_length
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(rng_seed)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            all_rngs = [stream for stream in self.rng.spawn(worker_info.num_workers)]
            rng = all_rngs[worker_info.id]
        else:
            rng = self.rng
        while True:
            parities = [
                generate_fixed_parity(
                    self.total_sequence_length,
                    self.sequence_length,
                    stream
                ) for stream in rng.spawn(self.batch_size)
            ]
            yield torch.asarray(np.stack(parities, axis=0))


@torch.no_grad()
def do_validation(model, valid_dataloaders):
    valid_losses = {}
    for seq_len, dataloader in valid_dataloaders.items():
        loss = model(next(dataloader).squeeze().to('cuda:0'), return_type='loss')
        valid_losses[f'validation/{seq_len}'] = loss.item()
    return valid_losses



def train(model, optimizer, scheduler, num_steps, dataloader, valid_dataloaders):

    with trange(num_steps) as t:
        for i in t:
            data = next(dataloader).squeeze()
            optimizer.zero_grad()
            loss = model(data.to('cuda:0'), return_type='loss')
            loss.backward()
            optimizer.step()
            scheduler.step()

            msg = {'train_loss': loss.item()}

            if i % 100 == 0:
                valid_losses = do_validation(model, valid_dataloaders)
                msg.update(valid_losses)

            if i % 100 == 0:
                t.set_postfix(loss=loss.item(), valid_losses=valid_losses['validation/100'])
            
            wandb.log(msg)
            


def main(args):

    cfg = {
        "d_model": 128,
        "d_head": 32,
        "n_heads": 4,
        "d_mlp": 512,
        "n_ctx": 512,
        "n_layers": 1,
        "d_vocab": 4,
        "act_fn": "relu"
    }
    num_steps = 200_000
    num_warmup = 1_000
    seed = 100

    wandb.init(config=cfg, entity='dstander', project='rasp-parities')

    config = HookedTransformerConfig(**cfg)
    model = HookedTransformer(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=num_warmup)
    annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_steps - num_warmup), eta_min=1.0e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, annealing], milestones=[num_warmup])

    train_dataset = CumulativeParityDataset(512, 6, 32, 1024, seed)
    valid_lengths = [32, 40, 50, 60, 70, 80, 90, 100]
    valid_datasets = {i: CumulativeParityFixed(512, i, 512, i) for i in valid_lengths}
    dataloader = iter(DataLoader(train_dataset, num_workers=16, pin_memory=True, prefetch_factor=4))
    valid_dataloaders = {k: iter(DataLoader(v, num_workers=2, pin_memory=True)) for k, v in valid_datasets.items()}

    wandb.watch(model, log='all', log_freq=200)

    try:
        train(model, optimizer, scheduler, num_steps, dataloader, valid_dataloaders)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(None)

