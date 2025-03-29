import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from utility import load_data, reconstruction_SADloss, abu_similarity, hyperVca, load_HSI, plotAbundancesGT
from utility import plotAbundancesSimple, plotEndmembersAndGT
from net import Ours
import os
import time
import pandas as pd
import random
import matplotlib.pyplot as plt

start_time = time.time()

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load DATA
datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'synthetic': 'synthetic10',
                'dc': 'DC2',
                'moffett': 'moffett',
                'moni': 'moni20',
                'apex': 'apex'}
dataset = "moffett"
hsi = load_HSI("D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.gt.shape[0]
col = hsi.cols
line = hsi.rows
band_number = data.shape[1]
num_runs = 1

if dataset == "Samson":
    LR, EPOCH, batch_size = 1e-3, 1500, 1
    step_size, gamma = 245, 0.9
    a, b, c = 1, 0.01, 0.001
    w = 0.5
    k, p = 5, 2
    weight_decay_param = 1e-6
if dataset == "moffett":
    LR, EPOCH, batch_size = 1e-2, 1500, 1
    step_size, gamma = 240, 0.9
    a, b, c = 1, 0.01, 0.001
    w = 0.1
    k, p = 5, 2
    weight_decay_param = 1e-4
MSE = torch.nn.MSELoss(size_average=True)

end = []
end2 = []
abu = []

output_path = 'D:/study/方法代码尝试/Results'
method_name = 'DFFN'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

for run in range(1, num_runs + 1):
    print('Start training!', 'run:', run)
    abundance_GT = torch.from_numpy(hsi.abundance_gt)
    abundance_GT = torch.reshape(abundance_GT, (col * line, endmember_number))
    original_HSI = torch.from_numpy(data)
    original_HSI = torch.reshape(original_HSI.T, (band_number, col, line))
    abundance_GT = torch.reshape(abundance_GT.T, (endmember_number, col, line))
    # data 初始化 要将torch.from_numpy放入这个run循环中，否则会变回numpy然后报错，因为程序是基于tensor
    image = np.array(original_HSI)

    GT_endmember = hsi.gt.T

    """计算相似度增强后的高光谱图像"""
    Y = abu_similarity(original_HSI, w)
    Y = Y.float()

    # load data
    train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    net = Ours(band_number, endmember_number, col * col, k, p)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name

    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            x = x.cuda()
            Y = Y.cuda()
            net.train().cuda()
            endmem, abunda, re1, re2 = net(Y)

            re1 = torch.reshape(re1, (band_number, col, col))
            re1 = re1.unsqueeze(0)
            re2 = torch.reshape(re2, (band_number, col, col))
            abunda = torch.reshape(abunda, (endmember_number, col, col))

            abu_neg_error = torch.mean(torch.relu(-abunda))
            abu_sum_error = torch.mean((torch.sum(abunda, dim=0) - 1) ** 2)
            abu_loss = abu_neg_error + abu_sum_error

            de_loss = reconstruction_SADloss(re1, re2)
            re_loss = reconstruction_SADloss(x, re2)

            total_loss = a * (re_loss) + b * abu_loss + c * de_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print("Epoch:", epoch, "| loss: %.4f" % total_loss.cpu().data.numpy())
        scheduler.step()

    endmember_hat, en_abundance, re1, re2 = net(Y)

    en_abundance = torch.squeeze(en_abundance)

    en_abundance = torch.reshape(en_abundance, [endmember_number, col * line])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, line, endmember_number])
    abundance_GT = torch.reshape(abundance_GT, [endmember_number, col * line])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [line, col, endmember_number])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()

    endmember_hat = endmember_hat.cpu().detach().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = endmember_hat.T

    GT_endmember = GT_endmember.T

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': en_abundance,
                                                                              'E': endmember_hat, })

    plotAbundancesSimple(en_abundance, abundance_GT, abundance_path, abu)
    plotEndmembersAndGT(endmember_hat, GT_endmember, endmember_path, end)

    torch.cuda.empty_cache()

    print('-' * 70)
end_time = time.time()
end = np.reshape(end, (-1, endmember_number + 1))
abu = np.reshape(abu, (-1, endmember_number + 1))
dt = pd.DataFrame(end)
dt2 = pd.DataFrame(abu)
dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各端元SAD及mSAD运行结果.csv')
dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '参照丰度图'
plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
print('程序运行时间为:', end_time - start_time, 's')
