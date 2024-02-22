
import math
import s3fs
from tqdm import trange
import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.utils import lm_cross_entropy_loss
import wandb
from automatic_circuits.groups import SymmetricGroupGeneratorScratchpad


from concurrent.futures import ThreadPoolExecutor



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


def scratchpad_loss(logits, sequence, n):
    mask = ((sequence[:, 1:] >= n) | (sequence[:, 1:] < (2*n))) * 1.0
    totals = mask.sum(dim=-1, keepdims=True)
    losses = lm_cross_entropy_loss(logits, sequence, per_token=True)
    return ((losses * mask).sum(dim=-1) / totals).mean()


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
    logits = model(data, return_type='logits')
    loss = lm_cross_entropy_loss(logits, data)
    acc = acc_fn(logits, data, group.order).mean()
    valid_msg[f'loss/validation'] = loss.item()
    valid_msg[f'accuracy/validation'] = acc.item()
    return valid_msg



def train(model, optimizer, config, num_steps, group, bucket):

    msg = do_validation(model, group)
    wandb.log(msg)

    executor = ThreadPoolExecutor(max_workers=20)

    with trange(1, num_steps + 1) as t:
        for i in t:
            data = group.generate().to('cuda:0')
            optimizer.zero_grad()
            logits = model(data.to('cuda'))
            loss = scratchpad_loss(logits, data, group.order)
            loss.backward()
            optimizer.step()

            msg = {'train_loss': loss.item()}

            if i % 50 == 0:
                valid_losses = do_validation(model, group)
                msg.update(valid_losses)

            if i % 100 == 0:
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

    wandb.init(entity='dstander', project='s4-transformer')

    N = 4
    group_order = math.factorial(N)
    context = 128
    batch_size = 512
    num_steps = 100_000
    seed = 0
    path = f'S{N}-{seed}'
    bucket = f's3://automatic-circuits-01/{path}'

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
        train(model, optimizer, cfg, num_steps, data, bucket)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(None)

