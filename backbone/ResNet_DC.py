import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.gumbel_distribution import GumbelConcrete


class Decoupler(torch.nn.Module): # Domain Feature Decoupler (DFD)
    def __init__(self, size, num_channel=64, tau=0.1):
        super(Decoupler, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        self.DFD_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat, is_eval=False):
        attr_map = self.DFD_net(feat)

        mask = attr_map.reshape(attr_map.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelConcrete(tau=self.tau)(mask, is_eval=is_eval)
        gumbel_mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)

        ro_feat = feat * mask
        re_feat = feat * (1 - mask)

        return ro_feat, re_feat, gumbel_mask


class Corrector(nn.Module): # Domain Feature Corrector (DFC)
    def __init__(self, size, num_channel=64):
        super(Corrector, self).__init__()
        C, H, W = size

        self.DFC_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, re_feat, gumbel_mask):
        rec_feat = self.DFC_net(re_feat)
        rec_feat = rec_feat * (1 - gumbel_mask)
        rec_feat = re_feat + rec_feat

        return rec_feat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet_DC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, gum_tau=0.1, image_size=(32, 32), name='F2DC'):
        super(ResNet_DC, self).__init__()
        self.name = name
        self.in_planes = 64
        self.tau = gum_tau
        self.image_size = image_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.DFD = Decoupler(size=(512, int(self.image_size[0] / 8), 
                                           int(self.image_size[1] / 8)), tau=self.tau)
        self.DFC = Corrector(size=(512, int(self.image_size[0] / 8), 
                                            int(self.image_size[1] / 8)))
        self.aux = nn.Sequential(nn.Linear(512, num_classes))

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_eval=False):
        ro_outputs = []
        re_outputs = []
        rec_outputs = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)

        ro_feat, re_feat, gumbel_mask = self.DFD(out, is_eval=is_eval)
        ro_flatten = torch.nn.AdaptiveAvgPool2d(1)(ro_feat).reshape(ro_feat.shape[0], -1)
        re_flatten = torch.nn.AdaptiveAvgPool2d(1)(re_feat).reshape(re_feat.shape[0], -1)

        ro_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(ro_feat).reshape(ro_feat.shape[0], -1))
        ro_outputs.append(ro_out)
        re_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(re_feat).reshape(re_feat.shape[0], -1))
        re_outputs.append(re_out)

        rec_feat = self.DFC(re_feat, gumbel_mask)
        rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        rec_outputs.append(rec_out)

        out = ro_feat + rec_feat
        # print(out.shape)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        feat = out
        out = self.linear(out)

        return out, feat, ro_outputs, re_outputs, rec_outputs, ro_flatten, re_flatten


def resnet10_dc(num_classes=7, gum_tau=0.1):
    return ResNet_DC(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, tau=gum_tau, image_size=(128, 128))

def resnet10_dc_office(num_classes=10, gum_tau=0.1):
    return ResNet_DC(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, tau=gum_tau, image_size=(32, 32))

def resnet10_dc_digits(num_classes=10, gum_tau=0.1):
    return ResNet_DC(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, tau=gum_tau, image_size=(32, 32))