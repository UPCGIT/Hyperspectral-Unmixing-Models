import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import random
from helper import *
from network import A2SN
import logging
import os
import pandas as pd
import cv2
import time
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'synthetic': 'synthetic10',
                'dc': 'DC2',
                'berlin': 'berlin_test',
                'apex': 'apex_new',
                'moni': 'moni30',
                'houston': 'houston_test',
                'sub': 'sub1'}
dataset = "Samson"
num_runs = 1

if dataset == 'Samson':
    image_file = r'D:\毕设\Datasets\Samson.mat'
    P, L, col = 3, 156, 95
    pixel = col ** 2
    LR, EPOCH, batch_size = 1e-3, 1000, 1
    step_size, gamma = 45, 0.9
    weight_decay_param = 1e-4
    drop = 0.1
elif dataset == 'Urban':
    image_file = r'D:\毕设\Datasets\Urban4.mat'
    P, L, col = 4, 162, 307
    pixel = col ** 2
    LR, EPOCH, batch_size = 1e-3, 300, 1
    step_size, gamma = 50, 0.9
    weight_decay_param = 1e-3
    drop = 0.1
elif dataset == 'berlin':
    image_file = r'D:\毕设\Datasets\berlin_test.mat'
    P, L, col = 5, 111, 100
    pixel = col ** 2
    LR, EPOCH, batch_size = 5e-3, 1000, 1
    step_size, gamma = 45, 0.8
    weight_decay_param = 1e-3
    drop = 0.1
elif dataset == 'dc':
    image_file = r'D:\毕设\Datasets\DC2.mat'
    P, L, col = 5, 191, 290
    pixel = col ** 2
    LR, EPOCH, batch_size = 2e-3, 400, 1
    step_size, gamma = 45, 0.8
    weight_decay_param = 1e-3
    drop = 0.1
elif dataset == 'apex':
    image_file = r'D:\毕设\Datasets\apex_new.mat'
    P, L, col = 4, 285, 110
    pixel = col ** 2
    LR, EPOCH, batch_size = 6e-3, 600, 1
    step_size, gamma = 40, 0.5
    weight_decay_param = 1e-3
    drop = 0.1
elif dataset == 'moni':
    image_file = 'D:/毕设/Datasets/' + datasetnames[dataset] + '.mat'
    P, L, col = 5, 166, 130
    pixel = col ** 2
    LR, EPOCH, batch_size = 5e-4, 300, 1
    step_size, gamma = 50, 0.9
    weight_decay_param = 0
    drop = 0.1
elif dataset == 'houston':
    image_file = r'D:\毕设\Datasets\houston_test.mat'
    P, L, col = 4, 144, 105
    pixel = col ** 2
    LR, EPOCH, batch_size = 5e-4, 300, 1
    step_size, gamma = 35, 0.6
    weight_decay_param = 5e-5
    drop = 0.1
elif dataset == 'sub':
    image_file = r'D:\毕设\Datasets\sub1.mat'
    P, L, col = 9, 279, 200
    pixel = col ** 2
    LR, EPOCH, batch_size = 5e-4, 300, 1
    step_size, gamma = 35, 0.6
    weight_decay_param = 5e-5
    drop = 0.1
else:
    raise ValueError("Unknown dataset")

endm = []
abun = []
output_path = './Results'
method_name = 'try'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)

time_start = time.time()
for run in range(1, num_runs + 1):
    print('Start training!', 'run:', run)

    data = sio.loadmat(image_file)

    HSI = np.asarray(data["Y"], dtype=np.float32)
    HSI = torch.from_numpy(HSI)  # mixed abundance
    GT = np.asarray(data["S_GT"], dtype=np.float32)
    GT = torch.from_numpy(GT)  # true abundance
    M_true = np.asarray(data['GT'], dtype=np.float32)

    band_Number = HSI.shape[0]
    col, col, endmember_number = GT.shape
    pixel_number = col * col

    HSI = torch.reshape(HSI, (L, col, col))

    model = 'A2SN'
    if model == 'A2SN':
        net = A2SN(L, P)
    else:
        logging.error("So such model in our zoo!")

    # load data
    train_dataset = MyTrainData(img=HSI, gt=GT, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_loss = 1

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            net.train().cuda()

            abu, end, re, xo = net(x)

            re = torch.reshape(re, (L, col, col))
            abu = torch.reshape(abu, (P, col, col))
            reloss = reconstruction_SADloss(x, re)

            abu_neg_error = torch.mean(torch.relu(-abu))
            abu_sum_error = torch.mean((torch.sum(abu, dim=-1) - 1) ** 2)
            abu_error = abu_neg_error + abu_sum_error

            total_loss = reloss

            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            loss = total_loss.cpu().data.numpy()

            if epoch % 100 == 0:
                print(
                    "Epoch:", epoch,
                    "| loss: %.4f" % total_loss.cpu().data.numpy(),
                )

    net.eval()
    abu, end, re, xo = net(x)

    re = torch.reshape(re, (L, col * col))
    re = re.detach().cpu().numpy()
    end = end.squeeze(0)
    end = end.detach().cpu().numpy()
    abu = torch.reshape(abu.T, (col, col, P))
    abu = abu.detach().cpu().numpy()
    GT = GT.cpu().numpy()

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'E': end,})

    plotEndmembersAndGT(end.T, M_true, endmember_path, endm)
    plotAbundancesSimple(abu, GT, abundance_path, abun)

time_end = time.time()
print('程序运行时间为:', time_end - time_start)
end = np.reshape(endm, (-1, endmember_number + 1))
abu = np.reshape(abun, (-1, endmember_number + 1))
dt = pd.DataFrame(end)
dt2 = pd.DataFrame(abu)
dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各端元SAD及mSAD运行结果.csv')
dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '参照丰度图'
plotAbundancesGT(GT, abundanceGT_path)

