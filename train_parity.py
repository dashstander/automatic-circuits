
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.utils import lm_accuracy, lm_cross_entropy_loss
from tqdm import trange
import torch
import wandb

from automatic_circuits.groups.cyclic import CyclicGroupGeneratorScratchpad

def scratchpad_accuracy(logits, sequence, n):
    labels = sequence[:, 1:][torch.where(sequence[:, 1:] >= n)]
    preds = logits[torch.where(sequence < n)].argmax(dim=1)
    return 1.0 * (labels == preds.argmax(dim=-1))




@torch.no_grad()
def do_validation(model, group):
    valid_msg = {}
    data = group.generate().to('cuda')
    n = group.N
    #even_inds = torch.arange(2, data.shape[1], 2).to('cuda:0')
    logits = model(data, return_type='logits')
    loss = lm_cross_entropy_loss(logits, data)
    acc = scratchpad_accuracy(logits, data, n)
    valid_msg[f'loss/validation'] = loss.item()
    valid_msg[f'accuracy/validation'] = acc.mean().item()
    return valid_msg


def train(model, optimizer, config, num_steps, group):

    with trange(num_steps) as t:
        for i in t:
            data = group.generate()
            optimizer.zero_grad()
            loss = model(data.to('cuda:0'), return_type='loss')
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
            if i % 10000 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'config': config
                }, f'checkpoints/c8_transformer/{i}.pth')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'config': config}, f'checkpoints/c8_transformer/{i}.pth'
    )
            


def main(args):

    N = 8
    context = 128
    batch_size = 512

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
    num_steps = 100_000
    num_warmup = 500
    seed = 100

    wandb.init(config=cfg, entity='dstander', project='transformer-adder')

    config = HookedTransformerConfig(**cfg)
    model = HookedTransformer(config)

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
        train(model, optimizer, config, num_steps, dataset)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(None)

