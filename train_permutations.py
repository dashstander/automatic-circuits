
import math
from tqdm import trange
import torch
from torch.nn.functional import log_softmax
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.utils import lm_accuracy, lm_cross_entropy_loss
import wandb


from automatic_circuits.groups import SymmetricGroupGenerator, SymmetricGroupGeneratorScratchpad



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


@torch.no_grad()
def do_validation(model, group):
    valid_msg = {}
    data = group.generate().to('cuda')
    n = group.N
    #even_inds = torch.arange(2, data.shape[1], 2).to('cuda:0')
    logits = model(data, return_type='logits')
    loss = lm_cross_entropy_loss(logits, data)
    acc = acc_fn(logits, data, group.order).mean()
    valid_msg[f'loss/validation'] = loss.item()
    valid_msg[f'accuracy/validation'] = acc.item()
    return valid_msg

def train(model, optimizer, config, num_steps, group):

    with trange(num_steps) as t:
        for i in t:
            data = group.generate()
            optimizer.zero_grad()
            loss = model(data.to('cuda'), return_type='loss')
            loss.backward()
            optimizer.step()

            msg = {'train_loss': loss.item()}

            if i % 100 == 0:
                valid_losses = do_validation(model, group)
                msg.update(valid_losses)

            if i % 100 == 0:
                t.set_postfix(loss=loss.item())
            
            wandb.log(msg)
            if i % 10000 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config
                }, f'checkpoints/s4_transformer/{i}.pth')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config
    }, f'checkpoints/s4_transformer/{i}.pth')
            

def main(_):

    wandb.init(entity='dstander', project='mamba-s4')

    N = 4
    group_order = math.factorial(N)
    context = 128
    batch_size = 512
    num_steps = 500_000
    seed = 100

    cfg = {
        "d_model": 256,
        "d_head": 64,
        "n_heads": 4,
        "d_mlp": 1024,
        "n_ctx": context * 2 + 2,
        "n_layers": 1,
        "d_vocab": group_order * 2 + 1,
        "act_fn": "relu"
    }

    torch.manual_seed(seed)

    config = HookedTransformerConfig(**cfg)
    model = HookedTransformer(config)
    model.to('cuda:0')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1.0)
   
    data = SymmetricGroupGeneratorScratchpad(N, context, batch_size)

    wandb.watch(model, log='all', log_freq=200)

    try:
        train(model, optimizer, cfg, num_steps, data)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(None)

