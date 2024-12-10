import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import random
from helper import *
from network import A2SAN
import logging
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'Samson'
if dataset == 'Samson':
    image_file = r'D:\毕设\Datasets\Samson.mat'
    P, L, col = 3, 156, 95
    pixel = col**2
    LR, EPOCH, batch_size = 1e-3, 1000, 1
    step_size, gamma = 45, 0.9
    a, b = 1, 1.5e-6
    weight_decay_param = 1e-4
else:
    raise ValueError("Unknown dataset")

data = sio.loadmat(image_file)

HSI = torch.from_numpy(data["Y"])  # mixed abundance
GT = torch.from_numpy(data["S_GT"])  # true abundance
M_true = data['GT']

band_Number = HSI.shape[0]
col, col, endmember_number = GT.shape
pixel_number = col * col

HSI = torch.reshape(HSI, (L, col, col))

model = 'A2SAN'
if model == 'A2SAN':
    net = A2SAN(L, P)
else:
    logging.error("So such model in our zoo!")

MSE = torch.nn.MSELoss(size_average=True)

# load data
train_dataset = MyTrainData(img=HSI, gt=GT, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

best_loss = 1
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):

        scheduler.step()
        x = x.cuda()
        net.train().cuda()
        abu, end, re, xo = net(x)

        re = torch.reshape(re, (L, col, col))
        abu = torch.reshape(abu, (P, col, col))
        reloss = reconstruction_SADloss(x, re)
        # reloss = MSE(x, re)

        abu_neg_error = torch.mean(torch.relu(-abu))
        abu_sum_error = torch.mean((torch.sum(abu, dim=-1) - 1) ** 2)
        abu_error = abu_neg_error + abu_sum_error

        total_loss = a * reloss + b * abu_error
        # total_loss = reloss

        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()

        loss = total_loss.cpu().data.numpy()

        if loss < best_loss:
            state = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, "./SamsonA2SNbest_model.pth.tar")
            best_loss = loss

        if epoch % 100 == 0:
            print(
                "Epoch:",
                epoch,
                "| loss: %.4f" % total_loss.cpu().data.numpy(),
                "| loss: %.4f" % abu_error.cpu().data.numpy(),
            )

checkpoint = torch.load("./SamsonA2SNbest_model.pth.tar")
best_loss = checkpoint['best_loss']
loss = checkpoint['loss']
epoch = checkpoint['epoch']

net.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
abu, end, re, xo = net(x)

xo = xo.detach().cpu().numpy()
re = torch.reshape(re, (L, col * col))
re = re.detach().cpu().numpy()
end = end.squeeze(0)
end = end.detach().cpu().numpy()
abu = torch.reshape(abu.T, (col, col, P))
abu = abu.detach().cpu().numpy()
GT = GT.cpu().numpy()

import os

results = './Results'
if not os.path.exists(results):
    os.makedirs(results)
endmember_path = './Results/endmember'
abundance_path = './Results/abundance'
abundanceGT_path = './Results/abundance_GT'
plotEndmembersAndGT(end.T, M_true, endmember_path)
plotAbundancesSimple(abu, GT, abundance_path)
plotAbundancesGT(GT, abundanceGT_path)
print(end.shape)
print(abu.shape)
