import os
import sys 
import uuid
from tqdm import tqdm
import torch
import airbench
from elastic_airbench94 import InfiniteCifarLoader, train, make_net

# If you want full determinism, must explicitly do this since importing airbench turns it on.
#torch.backends.cudnn.benchmark = False

with open(sys.argv[0]) as f:
    code = f.read()

train_loader = airbench.CifarLoader('/tmp/cifar10', train=True)
train_labels = train_loader.labels
test_loader = airbench.CifarLoader('/tmp/cifar10', train=False)

model = torch.compile(make_net(), mode='max-autotune')

def save_outs(mask, key, aug_seed=None, order_seed=None, **kwargs):

    loader = InfiniteCifarLoader('/tmp/cifar10', train=True, batch_size=1000,
                                 aug=dict(flip=True, translate=2), altflip=False,
                                 aug_seed=aug_seed, order_seed=order_seed, subset_mask=mask)

    train_logits = []
    test_logits = []
    for i in range(50):
        train(model, loader, **kwargs)
        train_logits.append(airbench.infer(model, train_loader))
        test_logits.append(airbench.infer(model, test_loader))
    obj = dict(code=code, mask=mask, train_logits=torch.stack(train_logits), test_logits=torch.stack(test_logits))
    os.makedirs('logits/%s' % key, exist_ok=True)
    torch.save(obj, os.path.join('logits', key, str(uuid.uuid4())+'.pt'))

def convert_mask(mask, labels=train_labels):
    """ 
    input: mask of shape [10, 5000] where mask[c, i] corresponds
           to whether the ith example of class c should be kept.
    output: mask of shape [50000] which implements this desiderata.
    """
    ind = torch.arange(len(labels), device=labels.device)
    class_idxs = torch.stack([ind[labels == c] for c in range(10)])
    keep_idxs = class_idxs[mask]
    mask_out = torch.tensor([False]*50000, device=labels.device)
    mask_out[keep_idxs] = True
    return mask_out

def get_firstk(k):
    mask = torch.empty(10, 5000, dtype=torch.bool, device=train_labels.device).fill_(False)
    mask[:, :k] = True
    return mask


for _ in tqdm(range(2000)):
    mask = get_firstk(4000)
    mask[:, 4000:] = (torch.rand(10, 1000) < 0.5)
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10_first4000_random_masks')

n = 50

for _ in tqdm(range(n)):
    mask = get_firstk(5000)
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10')

for _ in tqdm(range(n)):
    mask = get_firstk(4000)
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10_first4000')

for _ in tqdm(range(n)):
    mask = get_firstk(4000)
    mask[:, 4000:4500] = True
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10_first4000_first500')

for _ in tqdm(range(n)):
    mask = get_firstk(4000)
    mask[:, 4500:5000] = True
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10_first4000_last500')

for _ in tqdm(range(n)):
    mask = get_firstk(4000)
    mask[3, :] = True
    mask[5, :] = True
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10_first4000_catdog')

n = 10

for _ in tqdm(range(n)):
    mask = get_firstk(5000)
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e11', epochs=11)

for _ in tqdm(range(n)):
    mask = get_firstk(5000)
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e12', epochs=12)

for _ in tqdm(range(n)):
    mask = get_firstk(5000)
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e15', epochs=15)

for _ in tqdm(range(n)):
    mask = get_firstk(5000)
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e20', epochs=20)
