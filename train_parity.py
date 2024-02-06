import numpy as np
from tqdm import trange
from transformer_lens import HookedTransformerConfig, HookedTransformer
import torch
from torch.nn.functional import log_softmax
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
    sep = np.array([2])
    prob = rng.beta(2, 2)
    bits = rng.choice(2, (total_seq_len - 1,), replace=True, p=[prob, 1. - prob])
    parities = np.concatenate([sep, np.cumsum(bits) % 2])
    bits = np.concatenate([sep, bits])
    return bits, parities


def generate_packed_parity(total_seq_length, min_seq_length=6, max_seq_length=30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sep = 2
    sequence_lengths = get_seq_lengths(total_seq_length, min_seq_length, max_seq_length, rng)
    assert sum(sequence_lengths) <= total_seq_length
    [bits, parities] = list(zip(*[
            generate_cum_parity(seq_len, stream) for seq_len, stream in zip(sequence_lengths, rng.spawn(len(sequence_lengths)))
    ]))
    bits = np.concatenate(bits)
    parities = np.concatenate(parities)
    diff = total_seq_length - len(parities)
    bits = np.pad(bits, (0, diff), mode='constant', constant_values=(sep, sep))
    parities = np.pad(parities, (0, diff), mode='constant', constant_values=(sep, sep))
    return bits, parities


def generate_fixed_parity(total_seq_length, seq_length=30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sep = 2
    sequence_lengths = [seq_length] * (total_seq_length // seq_length)
    assert sum(sequence_lengths) <= total_seq_length
    [bits, parities] = list(zip(*[
            generate_cum_parity(seq_len, stream) for seq_len, stream in zip(sequence_lengths, rng.spawn(len(sequence_lengths)))
    ]))
    bits = np.concatenate(bits)
    parities = np.concatenate(parities)
    diff = total_seq_length - len(parities)
    bits = np.pad(bits, (0, diff), mode='constant', constant_values=(sep, sep))
    parities = np.pad(parities, (0, diff), mode='constant', constant_values=(sep, sep))
    return bits, parities


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
            [bits, parities] = list(zip(*[
                generate_packed_parity(
                    self.total_sequence_length,
                    self.min_sequence_length,
                    self.max_sequence_length,
                    stream
                ) for stream in rng.spawn(self.batch_size)
            ]))
            yield torch.asarray(np.stack(bits, axis=0)), torch.asarray(np.stack(parities, axis=0))



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
            [bits, parities] = list(zip(*[
                generate_fixed_parity(
                    self.total_sequence_length,
                    self.sequence_length,
                    stream
                ) for stream in rng.spawn(self.batch_size)
            ]))
            yield torch.asarray(np.stack(bits, axis=0)), torch.asarray(np.stack(parities, axis=0))



def seq2seq_cross_entropy_loss(logits, tokens, ignore_token=2):
    log_probs = log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Not using offsets because we're predicting the same token position, new _sequence
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :, :].gather(
        dim=-1, index=tokens[..., :, None]
    )[..., 0]
    

    return -predicted_log_probs.mean()


def seq2seq_accuracy(logits, tokens):
    predicted_tok = logits.argmax(dim=-1)
    return (predicted_tok == tokens).to(torch.float32).mean()


@torch.no_grad()
def do_validation(model, valid_dataloaders):
    valid_msg = {}
    for seq_len, dataloader in valid_dataloaders.items():
        data, labels = next(dataloader)
        parities = labels.squeeze().to('cuda:0')
        logits = model(data.squeeze().to('cuda:0'), return_type='logits')
        loss = seq2seq_cross_entropy_loss(logits, parities)
        acc = seq2seq_accuracy(logits, parities)
        valid_msg[f'val_loss/{seq_len}'] = loss.item()
        valid_msg[f'val_acc/{seq_len}'] = acc.item()
    return valid_msg


def train(model, optimizer, config, num_steps, dataloader, valid_dataloaders):

    with trange(num_steps) as t:
        for i in t:
            data, labels = next(dataloader)
            optimizer.zero_grad()
            logits = model(data.squeeze().to('cuda:0'), return_type='logits')
            loss = seq2seq_cross_entropy_loss(logits, labels.squeeze().to('cuda:0'))
            loss.backward()
            optimizer.step()
            #scheduler.step()

            msg = {'train_loss': loss.item()}

            if i % 100 == 0:
                valid_losses = do_validation(model, valid_dataloaders)
                msg.update(valid_losses)

            if i % 100 == 0:
                t.set_postfix(loss=loss.item(), valid_losses=valid_losses['val_acc/512'])
            
            wandb.log(msg)
            if i % 10000 == 0:
                torch.save({'model': model.state_dict(), 'config': config}, f'checkpoints/{i}.pth')
    torch.save({'model': model.state_dict(), 'config': config}, f'checkpoints/{i}.pth')
            


def main(args):

    cfg = {
        "d_model": 128,
        "d_head": 32,
        "n_heads": 4,
        "d_mlp": 512,
        "n_ctx": 512,
        "n_layers": 1,
        "d_vocab": 3,
        "act_fn": "relu"
    }
    num_steps = 100_000
    num_warmup = 500
    seed = 100

    wandb.init(config=cfg, entity='dstander', project='rasp-parities')

    config = HookedTransformerConfig(**cfg)
    model = HookedTransformer(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    #warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=num_warmup)
    #annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_steps - num_warmup), eta_min=1.0e-6)
    #scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, annealing], milestones=[num_warmup])

    #train_dataset = CumulativeParityDataset(512, 64, 256, 512, seed)
    train_dataset = CumulativeParityFixed(512, 512, 512, seed)
    valid_lengths = [8, 16, 32, 64, 128, 256, 512]
    valid_datasets = {i: CumulativeParityFixed(512, i, 1024, i) for i in valid_lengths}
    dataloader = iter(DataLoader(train_dataset, num_workers=16, pin_memory=True, prefetch_factor=4))
    valid_dataloaders = {k: iter(DataLoader(v, num_workers=2, pin_memory=True)) for k, v in valid_datasets.items()}

    wandb.watch(model, log='all', log_freq=200)

    try:
        train(model, optimizer, config, num_steps, dataloader, valid_dataloaders)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(None)

