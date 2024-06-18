# elastic_airbench.py
# Variant of airbench94 which can be scaled in a perfectly elastic manner.
# Also removes label smoothing and TTA.
# And lookahead, and progressive freezing of whitening layer bias.
# Attains 92.99 mean accuracy in 1.58 seconds on an H100.

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import airbench

torch.backends.cudnn.benchmark = True

# We express the main training hyperparameters (batch size, learning rate, momentum, and weight decay)
# in decoupled form, so that each one can be tuned independently. This accomplishes the following:
# * Assuming time-constant gradients, the average step size is decoupled from everything but the lr.
# * The size of the weight decay update is decoupled from everything but the wd.
# In constrast, normally when we increase the (Nesterov) momentum, this also scales up the step size
# proportionally to 1 + 1 / (1 - momentum), meaning we cannot change momentum without having to re-tune
# the learning rate. Similarly, normally when we increase the learning rate this also increases the size
# of the weight decay, requiring a proportional decrease in the wd to maintain the same decay strength.
#
# The practical impact is that hyperparameter tuning is faster, since this parametrization allows each
# one to be tuned independently. See https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/.

hyp = {
    'opt': {
        'epochs': 10.0,
        'batch_size': 1000,
        'lr': 5.0,              # learning rate per 1024 examples -- 5.0 is optimal with no smoothing, 10.0 with smoothing.
        'momentum': 0.85,
        'weight_decay': 0.015,  # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,    # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.0,
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'scaling_factor': 1/9,
    },
}

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class InfiniteCifarLoader:
    """
    CIFAR-10 loader which constructs every input to be used during training during the call to __iter__.
    The purpose is to support cross-epoch batches (in case the batch size does not divide the number of train examples),
    and support stochastic iteration counts in order to preserve perfect linearity/independence.
    """

    def __init__(self, path, train=True, batch_size=500, aug=None, altflip=True,
                 aug_seed=None, order_seed=None, subset_mask=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location='cuda')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        assert train
        self.aug_seed = aug_seed
        self.order_seed = order_seed
        self.altflip = altflip
        self.subset_mask = subset_mask if subset_mask is not None else torch.tensor([True]*len(self.images)).cuda()

    def set_random_state(self, seed, state):
        if seed is None:
            # If we don't get a data seed, then make sure to randomize the state using independent generator, since
            # it might have already been set by the model seed.
            import random
            torch.manual_seed(random.randint(0, 2**63))
        else:
            seed1 = 1000 * seed + state # just don't do more than 1000 epochs or else there will be overlap
            torch.manual_seed(seed1)

    def __iter__(self):

        # Preprocess
        images0 = self.normalize(self.images)
        # Pre-randomly flip images in order to do alternating flip later.
        assert self.aug.get('flip', False)
        if self.altflip:
            self.set_random_state(self.aug_seed, 0)
            images0 = batch_flip_lr(images0)
        # Pre-pad images to save time when doing random translation
        pad = self.aug.get('translate', 0)
        assert pad > 0
        images0 = F.pad(images0, (pad,)*4, 'reflect')
        labels0 = self.labels

        # Iterate infinitely
        epoch = 0
        batch_size = self.batch_size

        num_examples = self.subset_mask.sum().item()
        current_pointer = num_examples
        batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
        batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

        while True:

            if len(batch_images) == batch_size:

                # If we have a full batch ready then just yield it and reset.
                assert len(batch_images) == len(batch_labels)
                yield (batch_images, batch_labels)

                batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
                batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

            else:

                # Otherwise, we need to generate more data to add to the batch.
                assert len(batch_images) < batch_size
                if current_pointer >= num_examples:
                    # If we already reached the end of the last epoch then we need to generate
                    # a new augmented epoch of data (using random crop and alternating flip).
                    epoch += 1

                    self.set_random_state(self.aug_seed, epoch)
                    images1 = batch_crop(images0, 32)
                    if self.altflip:
                        images1 = images1 if epoch % 2 == 0 else images1.flip(-1)
                    else:
                        images1 = batch_flip_lr(images1)

                    self.set_random_state(self.order_seed, epoch)
                    indices = torch.randperm(len(self.images), device=images0.device)

                    # The effect of doing subsetting in this manner is as follows. If the permutation wants to show us
                    # our four examples in order [3, 2, 0, 1], and the subset mask is [True, False, True, False],
                    # then we will be shown the examples [2, 0]. It is the subset of the ordering.
                    # The purpose is to minimize the interaction between the subset mask and the randomness.
                    # So that the subset causes not only a subset of the total examples to be shown, but also a subset of
                    # the actual sequence of examples which is shown during training.
                    indices_subset = indices[self.subset_mask[indices]]
                    images1 = images1[indices_subset]
                    labels1 = labels0[indices_subset]
                    current_pointer = 0

                # Now we are sure to have more data in this epoch remaining.
                # This epoch's remaining data is given by (images1[current_pointer:], labels1[current_pointer:])
                # We add more data to the batch, up to whatever is needed to make a full batch (but it might not be enough).
                remaining_size = batch_size - len(batch_images)
                batch_images = torch.cat([batch_images, images1[current_pointer:current_pointer+remaining_size]])
                batch_labels = torch.cat([batch_labels, labels1[current_pointer:current_pointer+remaining_size]])
                current_pointer += remaining_size

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

## The eigenvectors of the covariance matrix of 2x2 patches of CIFAR-10 divided by sqrt eigenvalues.
eigenvectors_scaled = torch.tensor([
 8.9172e-02,  8.9172e-02,  8.8684e-02,  8.8623e-02,  9.3872e-02,  9.3872e-02,  9.3018e-02,  9.3018e-02,
 9.0027e-02,  9.0027e-02,  8.9233e-02,  8.9172e-02,  3.8818e-01,  3.8794e-01,  3.9111e-01,  3.9038e-01,
-1.0767e-03, -1.3609e-03,  3.8567e-03,  3.2330e-03, -3.9087e-01, -3.9087e-01, -3.8428e-01, -3.8452e-01,
-4.8242e-01, -4.7485e-01,  4.5435e-01,  4.6216e-01, -4.6240e-01, -4.5557e-01,  4.8975e-01,  4.9658e-01,
-4.3311e-01, -4.2725e-01,  4.2285e-01,  4.2896e-01, -5.0781e-01,  5.1514e-01, -5.1562e-01,  5.0879e-01,
-5.1807e-01,  5.2783e-01, -5.2539e-01,  5.1904e-01, -4.6460e-01,  4.7070e-01, -4.7168e-01,  4.6240e-01,
-4.7290e-01, -4.7461e-01, -5.0635e-01, -5.0684e-01,  9.5410e-01,  9.5117e-01,  9.2090e-01,  9.1846e-01,
-4.7363e-01, -4.7607e-01, -5.0439e-01, -5.0586e-01, -1.2539e+00,  1.2490e+00,  1.2383e+00, -1.2354e+00,
-1.2637e+00,  1.2666e+00,  1.2715e+00, -1.2725e+00, -1.1396e+00,  1.1416e+00,  1.1494e+00, -1.1514e+00,
-2.8262e+00, -2.7578e+00,  2.7617e+00,  2.8438e+00,  3.9404e-01,  3.7622e-01, -3.8330e-01, -3.9502e-01,
 2.6602e+00,  2.5801e+00, -2.6055e+00, -2.6738e+00, -2.9473e+00,  3.0312e+00, -3.0488e+00,  2.9648e+00,
 3.9111e-01, -4.0063e-01,  3.7939e-01, -3.7451e-01,  2.8242e+00, -2.9023e+00,  2.8789e+00, -2.8008e+00,
 2.6582e+00,  2.3105e+00, -2.3105e+00, -2.6484e+00, -5.9336e+00, -5.1680e+00,  5.1719e+00,  5.9258e+00,
 3.6855e+00,  3.2285e+00, -3.2148e+00, -3.6992e+00, -2.4668e+00,  2.8281e+00, -2.8379e+00,  2.4785e+00,
 5.4062e+00, -6.2031e+00,  6.1797e+00, -5.3906e+00, -3.3223e+00,  3.8164e+00, -3.8223e+00,  3.3340e+00,
-8.0000e+00,  8.0000e+00,  8.0000e+00, -8.0078e+00,  9.7656e-01, -9.9414e-01, -9.8584e-01,  1.0039e+00,
 7.5938e+00, -7.5820e+00, -7.6133e+00,  7.6016e+00,  5.5508e+00, -5.5430e+00, -5.5430e+00,  5.5352e+00,
-1.2133e+01,  1.2133e+01,  1.2148e+01, -1.2148e+01,  7.4141e+00, -7.4180e+00, -7.4219e+00,  7.4297e+00,
]).reshape(12, 3, 2, 2)

def make_net(widths=hyp['net']['widths']):
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1']),
        ConvGroup(widths['block1'], widths['block2']),
        ConvGroup(widths['block2'], widths['block3']),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()

############################################
#                Training                  #
############################################

def train(model, train_loader,
          label_smoothing=hyp['opt']['label_smoothing'], epochs=hyp['opt']['epochs'],
          learning_rate=hyp['opt']['lr'], weight_decay=hyp['opt']['weight_decay'], momentum=hyp['opt']['momentum'],
          bias_scaler=hyp['opt']['bias_scaler'], model_seed=None):

    batch_size = train_loader.batch_size
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = learning_rate / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = weight_decay * batch_size / kilostep_scale
    lr_biases = lr * bias_scaler

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    # Calculate the total train steps as an unbiased stochastic rounding of the possibly-fractional total batches
    # For example, with epochs=10 and batch_size=1000, we normally will always get 500 steps.
    # But if we input a train_loader with 49500 examples, then we will get 495 steps, with some of them being
    # across-epochs. And if we input a train_loader with 49950 examples, then we get 499.5 steps, i.e., a 50%
    # probability of 500 steps (where the 500th would involve a random 500 examples from an "11th" epoch), and
    # a 50% probability of 499 steps (where the last epoch would have a random 500 examples missing).
    num_examples = train_loader.subset_mask.sum().item()
    batches_per_epoch = num_examples / batch_size
    total_batches = epochs * batches_per_epoch
    integral_steps = int(total_batches)
    fractional_steps = total_batches - integral_steps
    # TODO: This is still nondeterministic when the batch size doesn't divide N. Not a case I really care about.
    if torch.rand(1)[0] <= fractional_steps:
        total_train_steps = integral_steps + 1
    else:
        total_train_steps = integral_steps

    # Reinitialize the network from scratch - nothing is reused from previous runs besides the PyTorch compilation
    if model_seed is None:
        # If we don't get a model seed, then make sure to randomize the state using independent generator, since
        # it might have already been set by the data seed inside the loader.
        import random
        torch.manual_seed(random.randint(0, 2**63))
    else:
        torch.manual_seed(model_seed)
    reinit_net(model)
    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.2)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (step - warmup_steps) / warmdown_steps
            return 1.0 * (1 - frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    ####################
    #     Training     #
    ####################

    model.train()
    for inputs, labels in train_loader:

        outputs = model(inputs)
        loss = loss_fn(outputs, labels).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_steps += 1
        if current_steps == total_train_steps:
            break

