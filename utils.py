import math
import os
import shutil
import numpy as np
from scipy.stats import norm
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np



def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path,'model_best.pth.tar'))

def load_checkpoint(model, checkpoint):
    m_keys = list(model.state_dict().keys())

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        c_keys = list(checkpoint['state_dict'].keys())
        not_m_keys = [i for i in c_keys if i not in m_keys]
        not_c_keys = [i for i in m_keys if i not in c_keys]
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        c_keys = list(checkpoint.keys())
        not_m_keys = [i for i in c_keys if i not in m_keys]
        not_c_keys = [i for i in m_keys if i not in c_keys]
        model.load_state_dict(checkpoint, strict=False)

    print("--------------------------------------\n LOADING PRETRAINING \n")
    print("Not in Model: ")
    print(not_m_keys)
    print("Not in Checkpoint")
    print(not_c_keys)
    print('\n\n')

def get_cifar10_dataloaders(train_batch_size, test_batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                        0.2010)),
    ])

    transform_test =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                        0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True,
                                             transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True,
                                            transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    subset_idx = np.random.randint(0, len(trainset), size=10000)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(subset_idx))

    return trainloader, valloader, testloader

def get_cifar10_dataloaders_disjoint(train_batch_size, test_batch_size):
    #np.random.seed(0)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                        0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                        0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True,transform=transform_train)

    total_idx = np.arange(0,len(trainset))
    np.random.shuffle(total_idx)
    subset_idx = total_idx[:10000]
    _subset_idx = total_idx[~np.in1d(total_idx, subset_idx)]
    valloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(subset_idx))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(_subset_idx))

    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss
def att(args,bifpn):
    return Attention(args)


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.p = 2
        self.kd = KLLoss()
        self.alpha = args.alpha
        self.beta = args.beta

    def forward(self, o_s, o_t, g_s, g_t):
        loss = self.alpha * self.kd(o_s, o_t)
        loss += self.beta * sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])

        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class DepthConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, depth=1):
        super(DepthConvBlock, self).__init__()
        conv = []
        if kernel_size == 1:
            conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
        else:
            conv.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            ))
            for i in range(depth-1):
                conv.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False, groups=out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                ))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)