import os
import sys
import uuid
from tqdm import tqdm
import torch
import airbench
from elastic_airbench94 import InfiniteCifarLoader, train, make_net

with open(sys.argv[0]) as f:
    code = f.read()

train_loader = airbench.CifarLoader('/tmp/cifar10', train=True)
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

n = 4
mask = torch.tensor([True]*50000).cuda()
for aug_seed, order_seed in [(None, None), (None, 0), (0, None), (0, 0)]:
    for _ in tqdm(range(n)):
        save_outs(mask, 'exp1/elastic_e10_seed_aug%s_order%s' % (aug_seed, order_seed), aug_seed, order_seed, epochs=10)
    for _ in tqdm(range(n)):
        save_outs(mask, 'exp1/elastic_e12_seed_aug%s_order%s' % (aug_seed, order_seed), aug_seed, order_seed, epochs=12)
    for _ in tqdm(range(n)):
        save_outs(mask, 'exp1/elastic_e14_seed_aug%s_order%s' % (aug_seed, order_seed), aug_seed, order_seed, epochs=14)

