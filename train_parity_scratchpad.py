import s3fs
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.utils import lm_accuracy, lm_cross_entropy_loss
from tqdm import trange
import torch
import wandb

from automatic_circuits.groups.cyclic import CyclicGroupGeneratorScratchpad




fs = s3fs.S3FileSystem()



def scratchpad_loss(logits, sequence, n):
    mask = ((sequence >= n) | (sequence < (2*n - 1))) * 1.0
    losses = lm_cross_entropy_loss(logits, sequence, per_token=True)
    return (losses * mask).sum(dim=-1).mean()

def scratchpad_acc(logits, sequence, n):
    mask = ((sequence[:, 1:] >= n) | (sequence[:, 1:] < (2*n - 1))) * 1.0
    totals = mask.sum(dim=-1, keepdims=True)
    acc = lm_accuracy(logits, sequence, per_token=True) * 1.0
    return ((acc * mask).sum(dim=-1) / totals).mean()




"""
def scratchpad_accuracy(logits, sequence, n):
    label_mask = (sequence >= n) & (sequence != 2*n)
    pred_mask = (sequence < n) & (sequence != 2*n)
    labels = sequence[label_mask]
    preds = logits[pred_mask, :].argmax(dim=-1)
    return (1.0 * (labels == preds)).mean()

def batched_accuracy(logits, data, n):
    acc = torch.zeros((logits.shape[0],), device=logits.device)
    for i in range(logits.shape[0]):
        acc[i] = scratchpad_accuracy(logits[i], data[i], n)
    return acc
        

acc_fn = torch.compile(batched_accuracy)
"""




@torch.no_grad()
def do_validation(model, group):
    valid_msg = {}
    data = group.generate().to('cuda')
    n = group.order
    #even_inds = torch.arange(2, data.shape[1], 2).to('cuda:0')
    logits = model(data, return_type='logits')
    loss = scratchpad_loss(logits, data)
    acc = scratchpad_acc(logits, data, n)
    valid_msg[f'loss/validation'] = loss.item()
    valid_msg[f'accuracy/validation'] = acc.item()
    return valid_msg


def train(model, optimizer, config, num_steps, group, bucket):

    with trange(1, num_steps + 1) as t:
        for i in t:
            data = group.generate().to('cuda:0')
            optimizer.zero_grad()
            logits = model(data)
            loss = scratchpad_loss(logits, data, group.order)
            loss.backward()
            optimizer.step()
            #scheduler.step()

            msg = {'loss/train': loss.item()}

            if i % 100 == 0:
                valid = do_validation(model, group)
                msg.update(valid)

            if i % 100 == 0:
                t.set_postfix(loss=loss.item())
            
            wandb.log(msg)
            if i % 5 == 0:
                with fs.open(f'{bucket}/{i}.pth', mode='wb') as file:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 
                        'config': config
                    }, 
                    file)
            


def main(args):

    N = 2
    context = 128
    batch_size = 512
    seed = 0
    path = f'C{N}-{seed}'
    bucket = f's3://automatic-circuits-01/{path}'
    

    cfg = {
        "d_model": 256,
        "d_head": 64,
        "n_heads": 4,
        "d_mlp": 1024,
        "n_ctx": context * 2 + 2,
        "n_layers": 1,
        "d_vocab": N * 2 + 1,
        "act_fn": "relu"
    }
    num_steps = 20_000
    num_warmup = 500

    wandb.init(config=cfg, entity='dstander', project='transformer-parity')

    config = HookedTransformerConfig(**cfg)
    model = HookedTransformer(config)

    with fs.open(f'{bucket}/0.pth', mode='wb') as file:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'config': config
            }, 
            file
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1.0)
    #warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=num_warmup)
    #annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_steps - num_warmup), eta_min=1.0e-6)
    #scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, annealing], milestones=[num_warmup])

    #train_dataset = CumulativeParityDataset(512, 64, 256, 512, seed)
    dataset = CyclicGroupGeneratorScratchpad(context, N, batch_size)
    #valid_lengths = [8, 16, 32, 64]
    #dataloader = iter(DataLoader(train_dataset, num_workers=16, pin_memory=True, prefetch_factor=4))
    #valid_dataloaders = {k: iter(DataLoader(v, num_workers=2, pin_memory=True)) for k, v in valid_datasets.items()}

    wandb.watch(model, log='all', log_freq=200)

    try:
        train(model, optimizer, config, num_steps, dataset, bucket)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(None)

