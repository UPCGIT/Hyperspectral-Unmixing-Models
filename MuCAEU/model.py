import torch
import torch.nn as nn
import numpy as np
from math import ceil


def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)


def transconv11(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)


class multiStageUnmixing(nn.Module):
    def __init__(self, band_Number, endmember_number, drop_out, col):
        super(multiStageUnmixing, self).__init__()
        self.endmember_number = endmember_number
        self.col = col
        self.layer1 = nn.Sequential(
            conv33(band_Number + endmember_number, 96),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
        )
        self.downsampling22 = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.downsampling44 = nn.AvgPool2d(4, 4, ceil_mode=True)
        self.layer2 = nn.Sequential(
            conv33(band_Number + endmember_number, 96),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
        )

        self.layer3 = nn.Sequential(
            conv33(band_Number, 96),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
        )

        self.encodelayer = nn.Sequential(nn.Softmax())
        self.transconv = transconv11(endmember_number, endmember_number)

        self.decoderlayer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )

        self.decoderlayer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )

        self.decoderlayer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ),
        )

    def forward(self, x):
        # layer3  模型结构图上最下面一层
        downsampling44 = self.downsampling44(x)
        layer3out = self.layer3(downsampling44)

        en_result3 = self.encodelayer(layer3out)
        de_result3 = self.decoderlayer6(en_result3)

        translayer3 = nn.functional.interpolate(layer3out, (ceil(self.col/2), ceil(self.col/2)), mode="bilinear")
        translayer3 = self.transconv(translayer3)

        # layer2
        downsampling22 = self.downsampling22(x)
        layer2out = torch.cat((downsampling22, translayer3), 1)
        layer2out = self.layer2(layer2out)
        en_result2 = self.encodelayer(layer2out)
        de_result2 = self.decoderlayer5(en_result2)

        # layer1  模型结构图中最上面一层
        translayer2 = nn.functional.interpolate(layer2out, (self.col, self.col), mode="bilinear")
        translayer2 = self.transconv(translayer2)
        layer1out = torch.cat((x, translayer2), 1)
        layer1out = self.layer1(layer1out)

        # layer1 out
        en_result1 = self.encodelayer(layer1out)
        de_result1 = self.decoderlayer4(en_result1)

        return en_result1, de_result1, en_result2, de_result2, en_result3, de_result3, downsampling22, downsampling44


