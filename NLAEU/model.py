
from torch.nn import Module, Sequential, Linear, Conv2d, LeakyReLU, Sigmoid, MaxPool2d, ConvTranspose2d


class NLAEU(Module):
    def __init__(self, P, Channel):
        super(NLAEU, self).__init__()
        self.P = P
        self.Channel = Channel
        self.encoder = Sequential(
            Linear(self.Channel, 128),
            LeakyReLU(0.1),
            Linear(128, 64),
            LeakyReLU(0.1),
            Linear(64, 16),
            LeakyReLU(0.1),
            Linear(16, self.P)
        )

        self.decoder_linearpart = Sequential(
            Linear(self.P, self.Channel, bias=False),
        )

        self.decoder_nonlinearpart = Sequential(
            Linear(self.Channel, self.Channel, bias=True),
            Sigmoid(),
            Linear(self.Channel, self.Channel, bias=True)
        )

    def forward(self, x):
        x_latent = self.encoder(x)
        x_latent = x_latent.abs()
        x_latent = x_latent.t() / x_latent.sum(1)
        x_latent = x_latent.t()
        x_latent = x_latent
        x_linear = self.decoder_linearpart(x_latent)
        x = self.decoder_nonlinearpart(x_linear)
        return x, x_latent

    def get_endmember(self, x):
        endmember = self.decoder_linearpart(x)
        return endmember


if __name__ == '__main__':
    import torch
    # 生成随机数据
    x = torch.randn(1024, 198)
    # 创建模型实例
    model = NLAEU(4, 198)
    # 运行模型
    output, latent = model(x)
    # 打印输出和潜变量
    print('Output:', output.shape)
    print('Latent:', latent)