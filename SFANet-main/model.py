import torch
import torch.nn as nn
from math import ceil
import numpy as np
import torch.nn.functional as F
import math
import random


def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)


def _aggregate(gate, D, I, K, sort=True):
    if sort:
        _, ind = gate.sort(descending=True)
        gate = gate[:, ind[0, :]]

    U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(_kronecker_product(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate


def _kronecker_product(mat1, mat2):
    return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(mat1.size() + mat2.size())).permute(
        [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))


class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=True,
                 groups=1):
        super(DGConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('D', torch.eye(2))
        self.register_buffer('I', torch.ones(2, 2))

        self.groups = groups
        if groups > 1:
            self.register_buffer('group_mask',
                                 _kronecker_product(torch.ones(out_channels // groups, in_channels // groups),
                                                    torch.eye(groups)))

        if self.out_channels // self.in_channels >= 2:  # Group-up
            self.K = int(np.ceil(math.log2(in_channels)))  # U: [in_channels, in_channels]
            r = int(np.ceil(self.out_channels / self.in_channels))
            _I = _kronecker_product(torch.eye(self.in_channels), torch.ones(r, 1))
            self._I = nn.Parameter(_I, requires_grad=False)
        elif self.in_channels // self.out_channels >= 2:  # Group-down
            self.K = int(np.ceil(math.log2(out_channels)))  # U: [out_channels, out_channels]
            r = int(np.ceil(self.in_channels / self.out_channels))
            _I = _kronecker_product(torch.eye(self.out_channels), torch.ones(1, r))
            self._I = nn.Parameter(_I, requires_grad=False)
        else:
            # in_channels=out_channels, or either one is not the multiple of the other
            self.K = int(np.ceil(math.log2(max(in_channels, out_channels))))

        eps = 1e-8
        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.sort = sort

    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())
        self.gate.data = ((self.gate.org - 0).sign() + 1) / 2.
        U_regularizer = 2 ** (self.K + torch.sum(self.gate))
        gate = torch.stack((1 - self.gate, self.gate))
        self.gate.data = self.gate.org  # Straight-Through Estimator
        U, gate = _aggregate(gate, self.D, self.I, self.K, sort=self.sort)
        if self.out_channels // self.in_channels >= 2:  # Group-up
            U = torch.mm(self._I, U)
        elif self.in_channels // self.out_channels >= 2:  # Group-down
            U = U[:self.out_channels, :self.out_channels]
            U = torch.mm(U, self._I)

        U = U[:self.out_channels, :self.in_channels]
        if self.groups > 1:
            U = U * self.group_mask
        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)

        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        return x


class Exponential_Local_Attention_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(Exponential_Local_Attention_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        if self.inter_channels == 0:
            self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        col = x1.size(2)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)  # [1, 16, 10000]
        t2 = self.T2(x2).view(batch_size, self.inter_channels, -1)  # [1, 16, 10000]
        t1 = t1.permute(0, 2, 1)  # [1, 10000, 16]

        # Calculate Affinity Matrix
        Affinity_M = torch.matmul(t1, t2)  #

        # Apply exponentiation to enhance feature saliency
        Affinity_M = torch.exp(Affinity_M)

        Affinity_M = Affinity_M.view(col, col, col*col)

        LAP = F.avg_pool2d(Affinity_M, kernel_size=3, stride=1, padding=1)  # [1, 100, 10000]

        RMP = LAP.max(dim=0, keepdim=True)[0]  #
        CMP = LAP.max(dim=1, keepdim=True)[0].permute(1, 0, 2)  #

        F_concat = torch.cat([RMP, CMP], dim=1)  #

        # Apply Global Average Pooling on the concatenated result to reduce the 200 dimension
        GAP = torch.mean(F_concat, dim=1, keepdim=True)  #

        GAP = GAP.view(1, col, col).unsqueeze(0)  #

        # Integrate with x1
        x1 = x1 * GAP.expand_as(x1)  # Element-wise multiplication
        # x1 = torch.cat([x1, GAP], dim=1)  # Concatenate along the channel dimension

        return x1


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out


class SFANet(nn.Module):
    def __init__(self, band_Number, band_Number_Lidar, endmember_number):
        super(SFANet, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # For image a (7×7×input_channels) --> (7×7×planes_a[0])
        self.conv1_a = conv_bn_relu(band_Number, self.planes_a[0], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×input_channels2) --> (7×7×planes_b[0])
        self.conv1_b = conv_bn_relu(band_Number_Lidar, self.planes_b[1], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[0]) --> (7×7×planes_a[1])
        # self.conv2_a = conv_bn_relu(self.planes_a[1], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        self.conv2_a = conv_bn_relu(self.planes_a[0], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[0]) --> (7×7×planes_b[1])
        self.conv2_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=1, bias=True)

        self.ELAM = Exponential_Local_Attention_Module(in_channels=self.planes_a[2],
                                                       inter_channels=self.planes_a[2] // 2)

        self.FusionLayer = nn.Sequential(
            DGConv2d(self.planes_a[2], endmember_number, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(endmember_number),
            nn.ReLU(),
        )

        self.softmax = nn.Softmax(dim=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(endmember_number, band_Number, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.conv1_a(x1)
        x2 = self.conv1_b(x2)

        x1 = self.conv2_a(x1)
        x2 = self.conv2_b(x2)

        ss_x1 = self.ELAM(x1, x2)

        x = self.FusionLayer(ss_x1)

        abu = self.softmax(x)
        output = self.decoder(abu)

        return abu, output