from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        # block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CIFAR_ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        super(CIFAR_ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        # block = Bottleneck if depth >=44 else BasicBlock

        block = Bottleneck
        # block = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                                 512 * block.expansion]

        fix_inplanes = self.inplanes
        self.layer1_1 = self._make_layer(block, 64, n,)
        self.inplanes = fix_inplanes  ##reuse self.inplanes
        self.layer2_1 = self._make_layer(block, 64, n)

        fix_inplanes = self.inplanes
        self.layer1_2 = self._make_layer(block, 128, n, stride=2)
        self.inplanes = fix_inplanes  ##reuse self.inplanes
        self.layer2_2 = self._make_layer(block, 128, n, stride=2)

        fix_inplanes = self.inplanes
        self.layer1_3 = self._make_layer(block, 256, n, stride=2)
        self.inplanes = fix_inplanes  ##reuse self.inplanes
        self.layer2_3 = self._make_layer(block, 256, n, stride=2)

        fix_inplanes = self.inplanes
        self.layer1_4 = self._make_layer(block, 512, n, stride=2)
        self.inplanes = fix_inplanes  ##reuse self.inplanes
        self.layer2_4 = self._make_layer(block, 512, n, stride=2)

        self.avgpool = nn.AvgPool2d(8)

        self.classfier1_4 = nn.Linear(512 * block.expansion, num_classes)
        self.classfier2_4 = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, preact=False):
        fmap = []
        fuse = []
        out = self.conv1(x)
        out = self.bn1(out)
        #out=self.relu(out)

        out1_1 = self.layer1_1(out)
        out2_1 = self.layer1_1(out)
        x_1 = 0.5 * (out1_1 + out2_1)
        fuse.append(out1_1)
        fuse.append(out2_1)
        fuse.append(x_1)

        out1_2 = self.layer1_2(out1_1)
        out2_2 = self.layer1_2(out2_1)
        x_2 = 0.5 * (out1_2 + out2_2)
        fuse.append(out1_2)
        fuse.append(out2_2)
        fuse.append(x_2)

        out1_3 = self.layer1_3(out1_2)
        out2_3 = self.layer1_3(out2_2)
        x_3 = 0.5 * (out1_3 + out2_3)
        fuse.append(out1_3)
        fuse.append(out2_3)
        fuse.append(x_3)

        out1_4 = self.layer1_4(out1_3)
        out2_4 = self.layer1_4(out2_3)
        x_4 = 0.5 * (out1_4 + out2_4)
        fuse.append(out1_4)
        fuse.append(out2_4)
        fuse.append(x_4)

        fmap.append(out1_4)
        fmap.append(out2_4)

        out1 = F.adaptive_avg_pool2d(F.relu(out1_4), (1, 1))
        out1 = out1.view(out1.size(0), -1)
        out2= F.adaptive_avg_pool2d(F.relu(out2_4), (1, 1))
        out2 = out2.view(out2.size(0), -1)


        out1=self.classfier1_4(out1)
        out2=self.classfier2_4(out2)


        # out = F.adaptive_avg_pool2d(F.relu(out1_4), (1,1))
        # out = out.view(out.size(0), -1)
        #out = self.fc(out)
        if not preact:
             return [F.relu(out1_1), F.relu(out1_2), F.relu(out1_3), F.relu(out1_4)],\
                    [F.relu(out2_1), F.relu(out2_2), F.relu(out2_3), F.relu(out2_4)],out1,out2,fmap,fuse
        return [out1_1,out1_2,out1_3,out1_4],[out2_1,out2_2,out2_3,out2_4],out1,out2,fmap,fuse


class Fusion_module(nn.Module):
    def __init__(self,channel,numclass,sptial):
        super(Fusion_module, self).__init__()
        self.fc2   = nn.Linear(channel, numclass)
        self.conv1 =  nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=1, padding=1, groups=channel*2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)


        self.sptial = sptial


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.avg = channel
    def forward(self, x,y):
        bias = False
        atmap = []
        input = torch.cat((x,y),1)

        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))

        atmap.append(x)
        x = F.avg_pool2d(x, self.sptial)
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        atmap.append(out)

        return out


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


def cifar_resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return CIFAR_ResNet(**kwargs)



















