import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import distLinear, mixup_data

act = torch.nn.ReLU()


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)



class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=200, loss_type='dist', per_img_std=False, stride=1,
                 dropRate=0.5):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        if loss_type == 'softmax':
            self.linear = nn.Linear(nChannels[3], int(num_classes))
            self.linear.bias.data.fill_(0)
        else:
            self.linear = distLinear(nChannels[3], int(num_classes))
        self.num_classes = num_classes

        self.feature_dim = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, target=None, mixup=False, mixup_hidden=True, lam=0.4, feature=True):
        if target is not None:
            if mixup_hidden:
                layer_mix = random.randint(0, 3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None

            out = x

            target_a = target_b = target

            if layer_mix == 0:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.conv1(out)
            out = self.block1(out)

            if layer_mix == 1:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.block2(out)

            if layer_mix == 2:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.block3(out)
            if layer_mix == 3:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)

            return out, out1, target_a, target_b
        else:
            out = x
            out = self.conv1(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)
            return out, out1


def wrn_s2m2(drop_rate=0.5, loss_type='dist', remove_linear=False, num_classes=64):
    if num_classes < 200:
        # Make the model compatible with the S2M2 checkpoints
        num_classes = 200

    model = WideResNet(depth=28, widen_factor=10, loss_type=loss_type, per_img_std=False, stride=1, dropRate=drop_rate,
                       num_classes=num_classes)
    return model
