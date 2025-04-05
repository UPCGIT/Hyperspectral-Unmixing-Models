import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChannelAttention(nn.Module):
    def __init__(self, kernel_size):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.LeakyReLU()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        m = self.max_pool(x)
        m = self.conv(m.squeeze(-1).transpose(-1, -2))
        m = m.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y + m)

        return y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.LeakyReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)

class SSAF(nn.Module):
    def __init__(self, P, Channel, rCol, nCol, z_dim, M0):
        super().__init__()
        self.P = P
        self.Channel = Channel
        self.rCol = rCol
        self.nCol = nCol
        self.M0 = M0

        self.sa1 = SpatialAttention(1)
        self.sa3 = SpatialAttention(5)

        self.ca1 = ChannelAttention(1)
        self.ca3 = ChannelAttention(1)

        self.AE_first_block1 = nn.Sequential(
            nn.Conv2d(self.Channel, 32 * self.P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * self.P),
            nn.LeakyReLU(0.01)
        )
        self.AE_first_block2 = nn.Sequential(
            nn.Conv2d(32 * self.P, 16 * self.P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.AE_first_block3 = nn.Sequential(
            nn.Conv2d(16 * self.P, 4 * self.P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.AE_first_block4 = nn.Sequential(
            nn.Conv2d(4 * self.P, self.P, kernel_size=3, stride=1, padding=1),
        )

        self.AE_second_block1 = nn.Sequential(
            nn.Conv2d(self.Channel, 32 * self.P, kernel_size=1, stride=1),
            nn.BatchNorm2d(32 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.AE_second_block2 = nn.Sequential(
            nn.Conv2d(32 * self.P, 16 * self.P, kernel_size=1, stride=1),
            nn.BatchNorm2d(16 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.AE_second_block3 = nn.Sequential(
            nn.Conv2d(16 * self.P, 4 * self.P, kernel_size=1, stride=1),
            nn.BatchNorm2d(4 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.AE_second_block4 = nn.Sequential(
            nn.Conv2d(4 * self.P, self.P, kernel_size=1, stride=1),
        )


        self.AE_first_decoder = nn.Sequential(
            nn.Linear(self.P, self.Channel, bias=False),
        )

        self.AE_second_decoder = nn.Linear(self.P, self.Channel)

        self.fc1 = nn.Linear(Channel, 32 * P)
        self.bn1 = nn.BatchNorm1d(32 * P)

        self.fc2 = nn.Linear(32 * P, 16 * P)
        self.bn2 = nn.BatchNorm1d(16 * P)

        self.fc3 = nn.Linear(16 * P, 4 * P)
        self.bn3 = nn.BatchNorm1d(4 * P)

        self.fc4 = nn.Linear(4 * P, z_dim)
        self.fc5 = nn.Linear(4 * P, z_dim)

        self.fc6 = nn.Linear(Channel, 32 * P)
        self.bn6 = nn.BatchNorm1d(32 * P)

        self.fc7 = nn.Linear(32 * P, 16 * P)
        self.bn7 = nn.BatchNorm1d(16 * P)

        self.fc8 = nn.Linear(16 * P, 4 * P)
        self.bn8 = nn.BatchNorm1d(4 * P)

        self.fc9 = nn.Linear(4 * P, z_dim)
        self.fc10 = nn.Linear(4 * P, z_dim)

        self.fc11 = nn.Sequential(
            nn.Linear(Channel, 32 * P),
            nn.BatchNorm1d(32 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(32 * P, 16 * P),
            nn.BatchNorm1d(16 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(16 * P, 4 * P),
            nn.BatchNorm1d(4 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(4 * P, 4 * P),
            nn.BatchNorm1d(4 * P),
            nn.LeakyReLU(0.0),

            nn.Linear(4 * P, P),
            nn.Softmax(dim=1)
        )

        self.fc12 = nn.Linear(z_dim, P * 2)
        self.bn12 = nn.BatchNorm1d(P * 2)

        self.fc13 = nn.Linear(P * 2, P * P)
        self.bn13 = nn.BatchNorm1d(P * P)

        self.fc17 = nn.Linear(P * P, P * P)

        self.fc14 = nn.Linear(z_dim, P * 4)
        self.bn14 = nn.BatchNorm1d(P * 4)

        self.fc15 = nn.Linear(P * 4, 64 * P)
        self.bn15 = nn.BatchNorm1d(64 * P)

        self.fc16 = nn.Linear(64 * P, Channel * P)

    def encoder_s(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc2(h1)
        h1 = self.bn2(h1)
        h11 = F.leaky_relu(h1, 0.00)

        h1 = self.fc3(h11)
        h1 = self.bn3(h1)
        h1 = F.leaky_relu(h1, 0.00)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

    def encoder_d(self, x):
        h1 = self.fc6(x)
        h1 = self.bn6(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h11 = F.leaky_relu(h1, 0.00)

        h1 = self.fc8(h11)
        h1 = self.bn8(h1)
        h1 = F.leaky_relu(h1, 0.00)

        mu = self.fc9(h1)
        log_var = self.fc10(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)

        return mu + eps * std

    def decoder_s(self, s):
        h1 = self.fc12(s)
        h1 = self.bn12(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc13(h1)
        h1 = self.bn13(h1)
        h1 = F.leaky_relu(h1, 0.00)

        psi = self.fc17(h1)

        return psi

    def deocer_d(self, d):
        h1 = self.fc14(d)
        h1 = self.bn14(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc15(h1)
        h1 = self.bn15(h1)
        h1 = F.leaky_relu(h1, 0.00)

        dM = self.fc16(h1)
        return dM

    def decoder_em(self, psi, dM):
        M0 = self.M0.repeat(psi.shape[0], 1, 1)
        em = M0 @ psi + dM
        em = torch.sigmoid(em)
        return em

    def first_encoder(self, input):
        x = self.AE_first_block1(input)
        x = self.sa1(x) * x
        x = self.AE_first_block2(x)
        x = self.AE_first_block3(x)
        x = self.sa3(x) * x
        x = self.AE_first_block4(x)
        a = F.softmax(x, dim=1)
        a = a.permute(2, 3, 0, 1)
        first_a = a.reshape(self.rCol * self.nCol, self.P)

        return first_a

    def second_encoder(self, input):
        x = self.AE_second_block1(input)
        x = self.ca1(x) * x
        x = self.AE_second_block2(x)
        x = self.AE_second_block3(x)
        x = self.ca3(x) * x
        x = self.AE_second_block4(x)
        a = F.softmax(x, dim=1)
        a = a.permute(2, 3, 0, 1)
        second_a = a.reshape(self.rCol * self.nCol, self.P)

        return second_a

    def forward(self, input):
        first_a = self.first_encoder(input)
        inputs_1 = input.reshape(self.rCol * self.nCol, self.Channel)
        mu_s, var_s = self.encoder_s(inputs_1)
        mu_d, var_d = self.encoder_d(inputs_1)
        s = self.reparameterize(mu_s, var_s)
        d = self.reparameterize(mu_s, var_s)
        psi = self.decoder_s(s)
        dM = self.deocer_d(d)
        psi_tensor = psi.view([-1, self.P, self.P])
        dM_tensor = dM.view([-1, self.Channel, self.P])
        em_tensor = self.decoder_em(psi_tensor, dM_tensor)
        em_tensor = em_tensor.view([-1, self.P, self.Channel])
        first_a_1 = first_a.view([-1, 1, self.P])
        first_y = first_a_1 @ em_tensor
        first_y = torch.squeeze(first_y, dim=1)

        second_input = first_y.reshape(self.rCol, self.nCol, self.Channel).unsqueeze(0)
        second_input = second_input.permute(0, 3, 1, 2)

        second_a = self.second_encoder(second_input)

        second_a_1 = second_a.view([-1, 1, self.P])
        second_y = second_a_1 @ em_tensor
        second_y = torch.squeeze(second_y, dim=1)

        return first_a, second_a, first_y, second_y, em_tensor, mu_s, mu_d, var_s, var_d
