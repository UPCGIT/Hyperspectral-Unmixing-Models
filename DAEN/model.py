from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# P:端元数量 Channel:波段数量
class DAEN(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(DAEN, self).__init__()
        self.P = P
        self.Channel = Channel
        self.intermediate_dim1 = int(np.ceil(Channel * 1.2) + 5)
        self.intermediate_dim2 = int(max(np.ceil(Channel / 4), z_dim + 2) + 3)
        self.intermediate_dim3 = int(max(np.ceil(Channel / 10), z_dim + 1))

        # encoder z  fc1 -->fc5  VAE部分
        self.fc1 = nn.Linear(Channel, self.intermediate_dim1)
        self.fc2 = nn.Linear(self.intermediate_dim1, self.intermediate_dim2)
        self.fc3 = nn.Linear(self.intermediate_dim2, self.intermediate_dim3)
        self.fc4 = nn.Linear(self.intermediate_dim3, z_dim)
        self.fc5 = nn.Linear(self.intermediate_dim3, z_dim)

        # encoder a  丰度部分
        self.fc9 = nn.Linear(Channel, 32 * P)
        self.bn9 = nn.BatchNorm1d(32 * P)

        self.fc10 = nn.Linear(32 * P, 16 * P)
        self.bn10 = nn.BatchNorm1d(16 * P)

        self.fc11 = nn.Linear(16 * P, 4 * P)
        self.bn11 = nn.BatchNorm1d(4 * P)

        self.fc12 = nn.Linear(4 * P, 4 * P)
        self.bn12 = nn.BatchNorm1d(4 * P)

        self.fc13 = nn.Linear(4 * P, 1 * P)

        # decoder
        self.fc6 = nn.Linear(z_dim, self.intermediate_dim2)
        self.fc7 = nn.Linear(self.intermediate_dim2, self.intermediate_dim1)
        self.fc8 = nn.Linear(self.intermediate_dim1, Channel * P)

    def encoder_z(self, x):
        h1 = self.fc1(x)
        h1 = F.relu(h1)

        h1 = self.fc2(h1)
        h11 = F.relu(h1)

        h1 = self.fc3(h11)
        h1 = F.relu(h1)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

    def encoder_a(self, x):
        h1 = self.fc9(x)
        h1 = self.bn9(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc10(h1)
        h1 = self.bn10(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc11(h1)
        h1 = self.bn11(h1)
        h1 = F.leaky_relu(h1, 0.00)

        h1 = self.fc12(h1)
        h1 = self.bn12(h1)

        h1 = F.leaky_relu(h1, 0.00)
        h1 = self.fc13(h1)

        a = F.softmax(h1, dim=1)
        return a

    # VAE重参数化
    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def decoder(self, z):
        h1 = self.fc6(z)
        h1 = F.relu(h1)

        h1 = self.fc7(h1)
        h1 = F.relu(h1)

        h1 = self.fc8(h1)
        em = torch.sigmoid(h1)
        return em

    def forward(self, inputs):
        mu, log_var = self.encoder_z(inputs)
        a = self.encoder_a(inputs)

        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        em = self.decoder(z)

        em_tensor = em.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)
        em_tensor = torch.squeeze(em_tensor, dim=0)

        return y_hat, mu, log_var, a, em_tensor


if __name__ == '__main__':
    P, bands, z_dim = 5, 200, 4
    device = 'cpu'
    model = DAEN(P, bands, z_dim)
    input = torch.randn(10, bands)
    y_hat, mu, log_var, a, em = model(input)
    print(' shape of y_hat: ', y_hat.shape)
    print(mu.shape)
    print(log_var.shape)
    print(em.shape)
    print(a.shape)

"""if __name__ == '__main__': 这行代码通常用于Python脚本中，用于区分代码是用作主程序运行还是用作模块被其他程序调用。
在Python中，当一个脚本被运行时，Python会将特殊变量__name__设置为字符串'__main__'。通过使用 if __name__ == '__main__':
这个条件，你可以指定只有当脚本作为主程序运行时才应该执行的代码，而不是当它作为模块被导入时执行的代码。
当做模块导入时 if __name__ == '__main__': 块中的代码不会被执行。"""
