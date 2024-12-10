import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, num_endmembers, num_bands):
        super(AutoEncoder, self).__init__()
        self.P = num_endmembers
        self.L = num_bands
        self.encoder = nn.Sequential(
            nn.Conv2d(self.L, 128, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.5),
            nn.ReLU(),
            # nn.Conv3d()
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, self.P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(self.P, momentum=0.9),
        )

        self.encodelayer = nn.Sequential(nn.Softmax(dim=1))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(self.P, self.L, kernel_size=1, stride=1, bias=False),

        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(self.P, self.L, kernel_size=1, stride=1, bias=False),

        )

    def forward(self, x):
        abu_est1 = self.encoder(x)

        abu_est1 = self.encodelayer(abu_est1)
        re_result1 = self.decoder1(abu_est1)
        end = self.decoder1[0].weight
        abu_est2 = self.encoder(re_result1)

        abu_est2 = self.encodelayer(abu_est2)
        re_result2 = self.decoder2(abu_est2)
        return abu_est1, re_result1, abu_est2, re_result2, end

