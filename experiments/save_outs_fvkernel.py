import os
import sys 
import uuid
from tqdm import tqdm
import torch
import torchvision
import airbench
from elastic_airbench94 import InfiniteCifarLoader, train, make_net

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))
class Loader:
    def __init__(self, path, train=True):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            if 'cifar10' in path:
                dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
                images = torch.tensor(dset.data)
                labels = torch.tensor(dset.targets)
                torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)
            elif 'cifar100' in path:
                dset = torchvision.datasets.CIFAR100(path, download=True, train=train)
                images = torch.tensor(dset.data)
                labels = torch.tensor(dset.targets)
                torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)
            else:
                assert False
        data = torch.load(data_path, map_location='cuda')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        self.normalize = torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)

# If you want full determinism, must explicitly do this since importing airbench turns it on.
#torch.backends.cudnn.benchmark = False

with open(sys.argv[0]) as f:
    code = f.read()

train_loader = Loader('/tmp/cifar10', train=True)
train_labels = train_loader.labels
test_loader = Loader('/tmp/cifar10', train=False)
train_loader2 = Loader('/tmp/cifar100', train=True)
test_loader2 = Loader('/tmp/cifar100', train=False)

model = torch.compile(make_net(), mode='max-autotune')

def save_outs(mask, key, aug_seed=None, order_seed=None, **kwargs):

    loader = InfiniteCifarLoader('/tmp/cifar10', train=True, batch_size=1000,
                                 aug=dict(flip=True, translate=2), altflip=False,
                                 aug_seed=aug_seed, order_seed=order_seed, subset_mask=mask)

    train_logits = []
    test_logits = []
    train_logits2 = []
    test_logits2 = []
    for i in range(50):
        train(model, loader, **kwargs)
        train_logits.append(airbench.infer(model, train_loader))
        test_logits.append(airbench.infer(model, test_loader))
        train_logits2.append(airbench.infer(model, train_loader2))
        test_logits2.append(airbench.infer(model, test_loader2))
    obj = dict(code=code, mask=mask, train_logits=torch.stack(train_logits), test_logits=torch.stack(test_logits),
               train_logits2=torch.stack(train_logits2), test_logits2=torch.stack(test_logits2))
    os.makedirs('logits/%s' % key, exist_ok=True)
    torch.save(obj, os.path.join('logits', key, str(uuid.uuid4())+'.pt'))

n = 99
for _ in tqdm(range(n)):
    mask = torch.tensor([True]*50000).cuda()
    save_outs(mask, 'elastic_e10_fvk')

