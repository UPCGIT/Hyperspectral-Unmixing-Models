import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from model import multiStageUnmixing
from utility import load_HSI, hyperVca, load_data, EdgeLoss, reconstruction_SADloss
from utility import plotAbundancesGT, plotAbundancesSimple, plotEndmembersAndGT, reconstruct
import time
import os
import pandas as pd
import random

start_time = time.time()
"""
seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'synthetic': 'synthetic30',
                'dc': 'dc_test',
                'sim': 'sim1020',
                'berlin': 'berlin_test',
                'moni': 'moni30',
                'houston': 'houston_lidar',
                'moffett': 'moffett',
                'muffle': 'muffle'}
dataset = "muffle"
hsi = load_HSI("D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.abundance_gt.shape[2]
col = hsi.cols
band_number = data.shape[1]

batch_size = 1
EPOCH = 800
num_runs = 10

if dataset == "Jasper":
    alpha = 0.1
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.001
    step_size = EPOCH//30
    gamma = 0.8
    weight_decay = 1e-4
if dataset == "Samson":
    alpha = 0.1
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.03
    step_size = 30
    gamma = 0.8
    weight_decay = 1e-4
if dataset == "Urban":
    alpha = 0.4
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.02
    step_size = 40
    gamma = 0.5
    weight_decay = 1e-3
if dataset == "synthetic":
    alpha = 0.45
    beta = 0.005
    dloss = 0.006
    drop_out = 0.2
    learning_rate = 0.001
    step_size = 45
    gamma = 0.8
    weight_decay = 1e-3
if dataset == "dc":
    alpha = 0.4
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.004
    step_size = 40
    gamma = 0.4
    weight_decay = 1e-3
if dataset == "sim":
    alpha = 0.45
    beta = 0.005
    drop_out = 0.2
    learning_rate = 0.001
    step_size = 35
    gamma = 0.4
    weight_decay = 1e-4
if dataset == "berlin":
    alpha = 0.4
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.004
    step_size = 40
    gamma = 0.4
    weight_decay = 1e-3
if dataset == "moni":
    alpha = 0.2
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.002
    step_size = 50
    gamma = 0.8
    weight_decay = 1e-3
if dataset == "apex":
    alpha = 0.4
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.02
    step_size = 40
    gamma = 0.5
    weight_decay = 1e-3
if dataset == "houston":# 0.01,40,0.7,5e-4
    alpha = 0.1
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.01
    step_size = 40
    gamma = 0.7
    weight_decay = 1e-3
if dataset == "moffett":
    alpha = 0.4
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.01
    step_size = 30
    gamma = 0.5
    weight_decay = 1e-3
if dataset == "muffle":
    alpha = 0.4
    beta = 0.03
    drop_out = 0.2
    learning_rate = 0.01
    step_size = 40
    gamma = 0.5
    weight_decay = 1e-3
MSE = torch.nn.MSELoss(size_average=True)

end = []
abu = []
r = []
output_path = 'D:/毕设/Results'
method_name = 'MuCAEU'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

# train
for run in range(1, num_runs + 1):
    print('Start training!', 'run:', run)
    # data 初始化 要将torch.from_numpy放入这个run循环中，否则会变回numpy然后报错，因为程序是基于tensor的
    abundance_GT = torch.from_numpy(hsi.abundance_gt)

    abundance_GT = torch.reshape(abundance_GT, (col * col, endmember_number))
    original_HSI = torch.from_numpy(data)
    original_HSI = torch.reshape(original_HSI.T, (band_number, col, col))
    abundance_GT = torch.reshape(abundance_GT.T, (endmember_number, col, col))

    # VCA_endmember and GT
    VCA_endmember, _, _ = hyperVca(data.T, endmember_number)
    GT_endmember = hsi.gt.T
    endmember_init = torch.from_numpy(VCA_endmember).unsqueeze(2).unsqueeze(
        3).float()  # 多生成两个波段，波段 * 端元 * 1 * 1 作为解码器初始权重

    # load data
    train_dataset = load_data(img=original_HSI, transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    edgeLoss = EdgeLoss(endmember_number)

    net = multiStageUnmixing(band_number, endmember_number, drop_out, col).cuda()

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name

    # weight init
    def weights_init(m):
        nn.init.kaiming_normal_(net.layer1[0].weight.data)
        nn.init.kaiming_normal_(net.layer1[4].weight.data)
        nn.init.kaiming_normal_(net.layer1[8].weight.data)

        nn.init.kaiming_normal_(net.layer2[0].weight.data)
        nn.init.kaiming_normal_(net.layer2[4].weight.data)
        nn.init.kaiming_normal_(net.layer2[8].weight.data)

        nn.init.kaiming_normal_(net.layer3[0].weight.data)
        nn.init.kaiming_normal_(net.layer3[4].weight.data)
        nn.init.kaiming_normal_(net.layer3[8].weight.data)


    net.apply(weights_init)

    # decoder weight init by VCA
    model_dict = net.state_dict()
    model_dict["decoderlayer4.0.weight"] = endmember_init
    model_dict["decoderlayer5.0.weight"] = endmember_init
    model_dict["decoderlayer6.0.weight"] = endmember_init

    net.load_state_dict(model_dict)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            scheduler.step()
            x = x.cuda()
            net.train().cuda()

            en_abundance, reconstruction_result, en_abundance2, reconstruction_result2, en_abundance3, reconstruction_result3, x2, x3 = net(
                x)

            abundanceLoss = reconstruction_SADloss(x, reconstruction_result)

            abundanceLoss2 = reconstruction_SADloss(x2, reconstruction_result2)

            abundanceLoss3 = reconstruction_SADloss(x3, reconstruction_result3)

            MSELoss = MSE(x, reconstruction_result)
            MSELoss2 = MSE(x2, reconstruction_result2)
            MSELoss3 = MSE(x3, reconstruction_result3)

            edge1 = edgeLoss(en_abundance2, en_abundance3)
            edge2 = edgeLoss(en_abundance, en_abundance2)

            ALoss = abundanceLoss + abundanceLoss2 + abundanceLoss3
            BLoss = MSELoss + MSELoss2 + MSELoss3
            CLoss = edge1 + edge2

            total_loss = ALoss + (alpha * BLoss) + (beta * CLoss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                print("Epoch:", epoch, "| loss: %.4f" % total_loss.cpu().data.numpy())

    net.eval()

    en_abundance, reconstruction_result, en_abundance2, reconstruction_result2, en_abundance3, reconstruction_result3, x2, x3 = net(
        x)
    en_abundance = torch.squeeze(en_abundance)
    en_abundance = torch.reshape(en_abundance, [endmember_number, col * col])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, col, endmember_number])
    abundance_GT = torch.reshape(abundance_GT, [endmember_number, col * col])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [col, col, endmember_number])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()

    endmember_hat = net.state_dict()["decoderlayer4.0.weight"].cpu().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = endmember_hat.T
    GT_endmember = GT_endmember.T

    y_hat = reconstruct(en_abundance, endmember_hat)
    RE = np.sqrt(np.mean(np.mean((y_hat - data) ** 2, axis=1)))
    r.append(RE)

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': en_abundance,
                                                                              'E': endmember_hat})

    plotAbundancesSimple(en_abundance, abundance_GT, abundance_path, abu)
    plotEndmembersAndGT(endmember_hat, GT_endmember, endmember_path, end)

    print('-' * 70)
end_time = time.time()
end = np.reshape(end, (-1, endmember_number + 1))
abu = np.reshape(abu, (-1, endmember_number + 1))
dt = pd.DataFrame(end)
dt2 = pd.DataFrame(abu)
dt3 = pd.DataFrame(r)
dt.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各端元SAD及mSAD运行结果.csv')
dt2.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
dt3.to_csv(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '重构误差RE运行结果.csv')
abundanceGT_path = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '参照丰度图'
plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
print('程序运行时间为:', end_time - start_time, 's')
