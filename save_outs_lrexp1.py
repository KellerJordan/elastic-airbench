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
    # ASSIGN ALL CATS TO DOG LABEL
    loader.labels[loader.labels == 3] = 5

    train_logits = []
    test_logits = []
    for i in range(50):
        train(model, loader, **kwargs)
        train_logits.append(airbench.infer(model, train_loader))
        test_logits.append(airbench.infer(model, test_loader))
    obj = dict(code=code, mask=mask, train_logits=torch.stack(train_logits), test_logits=torch.stack(test_logits))
    obj['train_logits'] = obj['train_logits'].float().mean(0).half()
    obj['test_logits'] = obj['test_logits'].float().mean(0).half()
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


n = 100
for _ in tqdm(range(n)):
    mask = get_firstk(5000)
    mask[3, :4500] = False # remove 80% of cats
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10_lrexp_mask1_lr4', learning_rate=4.0)
    save_outs(mask, 'elastic_e10_lrexp_mask1_lr5', learning_rate=5.0)
    save_outs(mask, 'elastic_e10_lrexp_mask1_lr6', learning_rate=6.0)

for _ in tqdm(range(n)):
    mask = get_firstk(5000)
    mask[5, :4500] = False # remove 80% of dogs
    mask = convert_mask(mask)
    save_outs(mask, 'elastic_e10_lrexp_mask2_lr4', learning_rate=4.0)
    save_outs(mask, 'elastic_e10_lrexp_mask2_lr5', learning_rate=5.0)
    save_outs(mask, 'elastic_e10_lrexp_mask2_lr6', learning_rate=6.0)


