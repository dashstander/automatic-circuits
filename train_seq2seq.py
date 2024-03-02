

from concurrent.futures import ThreadPoolExecutor
import s3fs 
from tqdm import trange
import torch
from torch.nn.functional import log_softmax
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.utils import lm_accuracy, lm_cross_entropy_loss
import wandb


from automatic_circuits.groups import CyclicGroupGenerator


fs = s3fs.S3FileSystem()


def save_to_s3(weights, optimizer, config, rng, bucket, step):
    with fs.open(f'{bucket}/{step}.pth', mode='wb') as file:
        torch.save(
            {
                'model': weights,
                'optimizer': optimizer, 
                'config': config,
                'rng': rng
            }, 
            file
        )


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
    logits = model(data).logits
    predicted_tok = logits.argmax(dim=-1)
    correct = (predicted_tok == labels).to(torch.float32)
    #correct, acc = seq2seq_accuracy(logits, parities)
    num_correct = correct.cumsum(dim=-1)
    for seq_len in valid_lengths:
        acc = num_correct[:, seq_len - 1].mean() / seq_len
        valid_msg[f'val_acc/{seq_len}'] = acc.item()
    return valid_msg


@torch.no_grad()
def do_validation(model, group):
    valid_msg = {}
    data, labels = [tensor.to('cuda') for tensor in group.generate()]
    #even_inds = torch.arange(2, data.shape[1], 2).to('cuda:0')
    logits = model(data, return_type='logits')
    loss = seq2seq_cross_entropy_loss(logits, labels)
    acc = seq2seq_accuracy(logits, labels)
    valid_msg[f'loss/validation'] = loss.item()
    valid_msg[f'accuracy/validation'] = acc.mean().item()
    return valid_msg


def train(model, optimizer, config, num_steps, group, bucket):

    msg = do_validation(model, group)
    wandb.log(msg)

    executor = ThreadPoolExecutor(max_workers=20)

    with trange(1, num_steps+1) as t:
        for i in t:
            data, labels = [tensor.to('cuda') for tensor in group.generate()]
            optimizer.zero_grad()
            logits = model(data, return_type='logits')
            loss = seq2seq_cross_entropy_loss(logits, labels)
            loss.backward()
            optimizer.step()

            msg = {'loss/train': loss.item()}

            if i % 10 == 0:
                valid_losses = do_validation(model, group)
                msg.update(valid_losses)

            if i % 10 == 0:
                t.set_postfix(loss=loss.item())
            
            wandb.log(msg)
            if i % 5 == 0:
                executor.submit(
                    save_to_s3,
                    model.state_dict(),
                    optimizer.state_dict(),
                    config,
                    torch.random.get_rng_state(),
                    bucket,
                    i
                )
            


def main(_):

    wandb.init(entity='dstander', project='mamba-parities')

    N = 2
    context = 32
    batch_size = 1024
    seed = 0
    path = f'C{N}-seq2seq-{seed}'
    bucket = f's3://automatic-circuits-01/{path}'
    

    cfg = {
        "d_model": 256,
        "d_head": 64,
        "n_heads": 4,
        "d_mlp": 1024,
        "n_ctx": context,
        "n_layers": 2,
        "d_vocab": N,
        "act_fn": "relu"
    }
    num_steps = 100_000

    torch.manual_seed(seed)

    wandb.init(config=cfg, entity='dstander', project='transformer-parity-seq2seq')

    config = HookedTransformerConfig(**cfg)
    model = HookedTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)


    with fs.open(f'{bucket}/0.pth', mode='wb') as file:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'config': config,
                'rng': torch.random.get_rng_state()
            }, 
            file
        )

    
    dataset = CyclicGroupGenerator(context, N, batch_size)
    
    wandb.watch(model, log='all', log_freq=200)

    try:
        train(model, optimizer, config, num_steps, dataset, bucket)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main(None)

