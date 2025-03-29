from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax


class Abu(nn.Module):
    def __init__(self, L, P, k, p):
        super(Abu, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=k, stride=1, padding=p),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=k, stride=1, padding=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(64, P, kernel_size=k, stride=1, padding=p),
            nn.BatchNorm2d(P),
            nn.ReLU(),
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=k, stride=1, padding=p),
            nn.BatchNorm2d(L),
            nn.ReLU(),
        )

    def forward(self, x):
        x11 = self.conv11(x)

        x12 = self.conv12(x11)

        abuT = self.conv13(x12)

        re1 = self.conv14(abuT)

        n, d, h, w = abuT.size()
        abuT = abuT.view(n, d, -1)
        abuT = abuT.permute(0, 2, 1)
        abu = abuT.permute(0, 2, 1)
        return abu, re1


class End(nn.Module):
    def __init__(self, P, Np):
        super(End, self).__init__()
        self.fc1 = nn.Linear(Np, 100)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, P)

    def forward(self, x):
        L = x.shape[1]
        Np = x.shape[2] * x.shape[3]

        x = torch.reshape(x, (L, Np))
        x1 = self.sigmoid(self.fc1(x))

        x2 = self.sigmoid(self.fc2(x1))

        x3 = self.sigmoid(self.fc3(x2))
        return x3


class Ours(nn.Module):
    def __init__(self, L, P, Np, k, p):
        super(Ours, self).__init__()
        self.Abu = Abu(L, P, k, p)
        self.End = End(P, Np)

    def forward(self, Y):
        abu, re1 = self.Abu(Y)

        end = self.End(re1)

        re2 = end @ abu
        return end, abu, re1, re2
