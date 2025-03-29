import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from utility import load_data, reconstruction_SADloss, SparseLoss, hyperVca, load_HSI, plotAbundancesGT
from utility import plotAbundancesSimple, plotEndmembersAndGT
from model import SFANet
import os
import time
import pandas as pd
import random
import matplotlib.pyplot as plt

start_time = time.time()
"""
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
"""
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load DATA
datasetnames = {'muffle': 'muffle',
                'houston': 'houston_lidar'
                }
dataset = "houston"
hsi = load_HSI("D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.gt.shape[0]
col = hsi.cols
line = hsi.rows
band_number = data.shape[1]
batch_size = 64
EPOCH = 500
num_runs = 1
dsm = hsi.dsm.transpose(2, 1, 0)
band_number_lidar = dsm.shape[0]

if dataset == "muffle":
    alpha = 0.1
    beta = 0.01
    lamda = 0.05
    drop_out = 0.3
    learning_rate = 0.006
if dataset == "houston":
    alpha = 0.5
    beta = 0.8
    lamda = 0.1
    drop_out = 0.3
    learning_rate = 0.01
MSE = torch.nn.MSELoss(size_average=True)

end = []
end2 = []
abu = []

output_path = 'D:/study/方法代码尝试/Results'
method_name = 'SFANet'
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

    endmembers, _, _ = hyperVca(data.T, endmember_number, datasetnames[dataset])
    VCA_endmember = torch.from_numpy(endmembers)
    GT_endmember = hsi.gt.T
    endmember_init = VCA_endmember.unsqueeze(2).unsqueeze(3).float()  # 多生成两个波段，波段 * 端元 * 1 * 1 作为解码器初始权重"""

    # load data
    train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    net = SFANet(band_number, band_number_lidar, endmember_number).cuda()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    endmember_path2 = endmember_folder + '/' + endmember_name + 'vca'

    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name

    # decoder weight init by VCA
    model_dict = net.state_dict()
    model_dict["decoder.0.weight"] = endmember_init
    net.load_state_dict(model_dict)

    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)

    dsm1 = torch.from_numpy(dsm).unsqueeze(0).to(device)
    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            x = x.cuda()
            net.train().cuda()

            en_abundance, reconstruction_result = net(x, dsm1)

            abundanceLoss = reconstruction_SADloss(x, reconstruction_result)
            MSELoss = MSE(x, reconstruction_result)
            L21Loss = SparseLoss(lamda)
            ALoss = abundanceLoss
            BLoss = MSELoss
            CLoss = L21Loss(en_abundance)
            total_loss = (alpha * ALoss) + (beta * BLoss) + CLoss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print("Epoch:", epoch, "| loss: %.4f" % total_loss.cpu().data.numpy())
        scheduler.step()
    en_abundance, reconstruction_result = net(x, dsm1)
    en_abundance = torch.squeeze(en_abundance)

    en_abundance = torch.reshape(en_abundance, [endmember_number, col * line])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, line, endmember_number])
    abundance_GT = torch.reshape(abundance_GT, [endmember_number, col * line])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [line, col, endmember_number])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()

    endmember_hat = net.state_dict()["decoder.0.weight"].cpu().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = endmember_hat.T

    GT_endmember = GT_endmember.T

    VCA_endmember = VCA_endmember.cpu().numpy()
    plotEndmembersAndGT(VCA_endmember.T, GT_endmember, endmember_path2, end2)

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': en_abundance,
                                                                              'E': endmember_hat,})

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

