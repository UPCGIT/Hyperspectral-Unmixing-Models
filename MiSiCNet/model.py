from utility import conv
import torch.nn as nn
import torch


class MiSiCNet(nn.Module):
    def __init__(self, input_depth, need_bias, pad, rmax, p1, nr1, nc1):
        super(MiSiCNet, self).__init__()
        self.rmax = rmax
        self.nr1 = nr1
        self.nc1 = nc1
        self.conv1 = nn.Sequential(
            conv(input_depth, 256, 3, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv(256, 256, 3, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.skipconv = nn.Sequential(
            conv(input_depth, 4, 1, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=1),
            conv(260, 256, 3, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Upsample(scale_factor=1),
            conv(256, rmax, 3, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(rmax, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        )

        self.encodelayer = nn.Sequential(nn.Softmax())
        self.dconv = nn.Sequential(
            nn.Linear(rmax, p1, bias=False),
        )

    def forward(self, x):
        x1 = self.skipconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([x, x1], 1)
        x = self.conv3(x)
        x2 = self.conv4(x)
        x2 = self.encodelayer(x2)
        x3 = torch.transpose(x2.view((self.rmax, self.nr1 * self.nc1)), 0, 1)
        x3 = self.dconv(x3)
        return x2, x3
