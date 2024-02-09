from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import numpy as np
from tqdm import trange
import torch
from torch.nn.functional import log_softmax
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Dirichlet, Categorical

import wandb


def generate_cum_addition(seq_len: int, n: int, batch_size: int):
    summands = torch.empty((batch_size, seq_len), dtype=torch.int32)
    for i in range(batch_size): 
        probs = Dirichlet(torch.ones(n,)).sample()
        dist = Categorical(probs)
        summands[i, :] = dist.sample((seq_len,))
    sums = summands.cumsum(dim=-1) % n
    return summands, sums



class CumulativeAdditionGenerator:

    def __init__(self, seq_len: int, N: int, batch_size: int, device):
        self.seq_len = seq_len
        self.N = N
        self.batch_size = batch_size
        self.device = device
        self.gen_fn = torch.compile(generate_cum_addition)

    def generate(self):
        summands, sums = self.gen_fn(self.seq_len, self.N, self.batch_size)
        return summands.to(self.device), sums.to(self.device)


def seq2seq_cross_entropy_loss(logits, tokens):
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
    correct = (predicted_tok == tokens).to(torch.float32)
    return correct, correct.mean()


@torch.no_grad()
def do_validation(model, dataloader, valid_lengths):
    valid_msg = {}
    data, labels = dataloader.generate()
    parities = labels
    logits = model(data).logits
    predicted_tok = logits.argmax(dim=-1)
    correct = (predicted_tok == labels).to(torch.float32)
    #correct, acc = seq2seq_accuracy(logits, parities)
    num_correct = correct.cumsum(dim=-1)
    for seq_len in valid_lengths:
        acc = num_correct[:, seq_len - 1].mean() / seq_len
        valid_msg[f'val_acc/{seq_len}'] = acc.item()
    return valid_msg


def train(model, optimizer, config, num_steps, dataloader, valid_lengths):

    with trange(num_steps) as t:
        for i in t:
            data, labels = dataloader.generate()
            optimizer.zero_grad()
            logits = model(data).logits
            loss = seq2seq_cross_entropy_loss(logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            msg = {'train_loss': loss.item()}

            if i % 100 == 0:
                valid_losses = do_validation(model, dataloader, valid_lengths)
                msg.update(valid_losses)

            if i % 100 == 0:
                t.set_postfix(loss=loss.item())
            
            wandb.log(msg)
            if i % 10000 == 0:
                torch.save({'model': model.state_dict(), 'config': config}, f'checkpoints/{i}.pth')
    torch.save({'model': model.state_dict(), 'config': config}, f'checkpoints/{i}.pth')
            


def main(_):

    wandb.init(entity='dstander', project='mamba-parities')

    ssm_config = {
        'd_state': 16,
        'd_conv': 2,
        'expand': 2
    }

    cfg = {
        'n_layer': 2,
        'd_model': 128,
        'vocab_size': 2,
        'rms_norm': True,
        'residual_in_fp32': True,
        'fused_add_norm':  True,
        'pad_vocab_size_multiple': 8,
        'ssm_cfg': ssm_config
    }

    num_steps = 500_000

    seed = 100
    torch.manual_seed(seed)

    config = MambaConfig(**cfg)
    model = MambaLMHeadModel(config, device='cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.)
   
    valid_lengths = [8, 16, 32, 64]
    dataloader = CumulativeAdditionGenerator(64, 2, 512, 'cuda:0')

    wandb.watch(model, log='all', log_freq=200)

    try:
        train(model, optimizer, cfg, num_steps, dataloader, valid_lengths)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(None)

