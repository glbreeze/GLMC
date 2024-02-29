import torch.nn as nn
import torch
import random
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from .utils import *


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# ======================== modified ResNet ========================
class ResNet_modify(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100, nf=64, etf_cls=False, fnorm='none'):
        super(ResNet_modify, self).__init__()
        self.in_planes = nf
        self.num_classes = num_classes
        self.etf_cls = etf_cls
        self.fnorm = fnorm

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.out_dim = 4 * nf * block.expansion
        if fnorm == 'nn1':
            self.bn4 = nn.BatchNorm1d(self.out_dim, affine=False)
            bias = False
        elif fnorm == 'nn2':  # batch norm, normalize feature
            self.bn4 = nn.BatchNorm1d(self.out_dim)
            self.fc5 = nn.Linear(self.out_dim, self.out_dim)
            self.bn5 = nn.BatchNorm1d(self.out_dim, affine=False)
            bias = False
        elif fnorm == 'none' or fnorm == 'null':
            bias = True

        self.fc = nn.Linear(self.out_dim, num_classes, bias=bias)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)

        self.apply(_weights_init)

        if etf_cls:
            weight = torch.sqrt(torch.tensor(num_classes / (num_classes - 1))) * (
                    torch.eye(num_classes) - (1 / num_classes) * torch.ones((num_classes, num_classes)))
            weight /= torch.sqrt((1 / num_classes * torch.norm(weight, 'fro') ** 2))  # [K, K]

            self.fc.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, self.out_dim)))  # [K, d]
            self.fc.weight.requires_grad_(False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, ret=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)
        if self.fnorm == 'nn1':
            feature = self.bn4(feature)
        elif self.fnorm == 'nn2':
            feature = self.bn4(feature)
            feature = self.fc5(feature)
            feature = self.bn5(feature)

        if self.fnorm == 'nn1' or self.fnorm == 'nn2':
            feature = F.normalize(feature, p=2, dim=-1)

        if ret is None:
            out = self.fc_cb(feature)
            return out
        elif ret == 'of':
            out = self.fc(feature)
            return out, feature
        else:
            out = self.fc(feature)
            return out

    def forward_mixup(self, x, target=None, mixup=None, mixup_alpha=None):

        if mixup >= 0 and mixup <= 3:
            layer_mix = mixup
        elif mixup == 9:
            layer_mix = random.randint(0, 3)
        else:
            layer_mix = None

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.tensor([lam], dtype=torch.float32, device=x.device)
            lam = torch.autograd.Variable(lam)

        target = to_one_hot(target, self.num_classes)
        if layer_mix == 0:
            x, target = mixup_process(x, target, lam)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        if layer_mix == 1:
            x, target = mixup_process(x, target, lam)

        x = self.layer2(x)
        if layer_mix == 2:
            x, target = mixup_process(x, target, lam)

        x = self.layer3(x)
        if layer_mix == 3:
            x, target = mixup_process(x, target, lam)

        feat = F.avg_pool2d(x, x.size()[3])
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)

        return out, target, feat


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_class=100, etf_cls=False):
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_class

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # from torch.nn import w

        self.fc = nn.Linear(512 * block.expansion, num_class)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, ret=None):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        feature = output.view(output.size(0), -1)
        if ret == 'of':
            out = self.fc(feature)
            return out, feature
        else:
            out = self.fc(feature)
            return out

    def forward_mixup(self, x, target=None, mixup=None, mixup_alpha=None):
        if mixup >= 0 and mixup <= 5:
            layer_mix = mixup
        elif mixup == 9:
            layer_mix = random.randint(0, 5)
        else:
            layer_mix = None

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.tensor([lam], dtype=torch.float32, device=x.device)
            lam = torch.autograd.Variable(lam)

        target = to_one_hot(target, self.num_classes)
        if layer_mix == 0:
            x, target = mixup_process(x, target, lam)

        x = self.conv1(x)
        if layer_mix == 1:
            x, target = mixup_process(x, target, lam)

        x = self.conv2_x(x)
        if layer_mix == 2:
            x, target = mixup_process(x, target, lam)

        x = self.conv3_x(x)
        if layer_mix == 3:
            x, target = mixup_process(x, target, lam)

        x = self.conv4_x(x)
        if layer_mix == 4:
            x, target = mixup_process(x, target, lam)

        x = self.conv5_x(x)
        if layer_mix == 5:
            x, target = mixup_process(x, target, lam)

        x = self.avg_pool(x)
        feat = x.view(x.size(0), -1)

        out = self.fc_cb(feat)
        return out, target, feat


# ======================== new ResNet ========================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IResNet(nn.Module):
    def __init__(self, block, layers, num_class=10, use_se=False):
        self.inplanes = 64
        self.use_se = use_se
        super(IResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 4 * 4, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc = nn.Linear(512, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x, ret='o'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc5(x)
        feature = self.bn5(x)

        if ret == 'of':
            out = self.fc(feature)
            return out, feature
        else:
            out = self.fc(feature)
            return out


def resnet18(num_class=100, etf_cls=False):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class=num_class, etf_cls=etf_cls)


def resnet32(num_class=10, etf_cls=False, fnorm='none'):
    return ResNet_modify(BasicBlock_s, [5, 5, 5], num_classes=num_class, etf_cls=etf_cls, fnorm=fnorm)


def resnet34(num_class=100, etf_cls=False):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class=num_class, etf_cls=etf_cls)


def resnet50(num_class=100, etf_cls=False):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_class=num_class, etf_cls=etf_cls)


def resnet101(num_class=100, etf_cls=False):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_class=num_class, etf_cls=etf_cls)


def resnet152(num_class=100, etf_cls=False):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_class=num_class, etf_cls=etf_cls)

def iresnet50(num_class=10, etf_cls=False, use_se=True, **kwargs):
    return IResNet(IRBlock, [3, 4, 14, 3], use_se=False, **kwargs)
