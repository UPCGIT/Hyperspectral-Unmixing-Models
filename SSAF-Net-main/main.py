import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from utility import load_data, reconstruction_SADloss, FCLSU, hyperVca, load_HSI, plotAbundancesGT
from utility import plotAbundancesSimple, plotEndmembersAndGT
from model import SSAF
import os
import time
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
start_time = time.time()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)


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
dataset = "Samson"
hsi = load_HSI("D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
endmember_number = hsi.gt.shape[0]
col = hsi.cols
line = hsi.rows
band_number = data.shape[1]
num_runs = 1
z_dim = 4
epochs = 2000
batch_size = 1

if dataset == "Samson":
    lr = 0.005
    lambda_y2 = 0.04
    lambda_kl = 0.001
    lambda_pre = 10
    lambda_sad = 5
    lambda_vol = 10
if dataset == "Jasper":
    lr = 0.005
    lambda_y2 = 0.04
    lambda_kl = 0.001
    lambda_pre = 10
    lambda_sad = 5
    lambda_vol = 10
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

    endmembers, _, _ = hyperVca(data.T, endmember_number, datasetnames[dataset])
    vca_em_l = endmembers.T
    M0 = np.reshape(vca_em_l, [1, vca_em_l.shape[1], vca_em_l.shape[0]]).astype('float32')
    M0 = torch.tensor(M0).to(device)

    FCLS_a = FCLSU(endmembers, data.T, 0.01)
    FCLS_a = FCLS_a.clone().detach()
    FCLS_a = torch.reshape(FCLS_a, (endmember_number, col, col)).unsqueeze(0)

    train_cube = torch.utils.data.TensorDataset(original_HSI.unsqueeze(0), FCLS_a)
    train_cube = torch.utils.data.DataLoader(train_cube, batch_size=batch_size, shuffle=True)

    model = SSAF(endmember_number, band_number, col, col, z_dim, M0).to(device)
    model.apply(weights_init)

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name

    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        for step, (y, fcls_a) in enumerate(train_cube):
            y = y[0].unsqueeze(0).to(device)
            fcls_a = fcls_a[0].reshape(endmember_number, col*col).to(device)
            first_a, second_a, first_y, second_y, em_tensor, mu_s, mu_d, var_s, var_d = model(y)
            y = y.permute(2, 3, 0, 1)
            y = y.reshape(col * col, band_number)

            loss_y_1 = ((first_y - y) ** 2).sum() / y.shape[0]
            loss_y_2 = ((y - second_y) ** 2).sum() / y.shape[0]

            loss_rec = loss_y_1 + lambda_y2 * loss_y_2

            loss_kl = -0.5 * (var_s + 1 - mu_s ** 2 - var_s.exp())
            loss_kl = loss_kl.sum() / y.shape[0]
            loss_kl = torch.max(loss_kl, torch.tensor(0.2).to(device))
            loss_a1_a2 = ((first_a - second_a) ** 2).sum() / first_a.shape[0]

            if epoch < epochs // 2:
                loss_a = (first_a.T - fcls_a).square().sum() / y.shape[0]
                loss = loss_rec + lambda_kl * loss_kl + lambda_pre * loss_a + 0.1 * loss_a1_a2

            else:
                em_bar = em_tensor.mean(dim=1, keepdim=True)
                loss_minvol = ((em_tensor - em_bar) ** 2).sum() / y.shape[0] / endmember_number / band_number

                em_bar = em_tensor.mean(dim=0, keepdim=True)
                aa = (em_tensor * em_bar).sum(dim=2)
                em_bar_norm = em_bar.square().sum(dim=2).sqrt()
                em_tensor_norm = em_tensor.square().sum(dim=2).sqrt()

                sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
                loss_sad = sad.sum() / y.shape[0] / endmember_number

                loss = loss_rec + lambda_kl * loss_kl + lambda_vol * loss_minvol + lambda_sad * loss_sad

            if epoch % 100 == 0:
                print("Epoch:", epoch, "| loss: %.4f" % loss.cpu().data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print("Epoch:", epoch, "| loss: %.4f" % loss.cpu().data.numpy())

    first_a, second_a, first_y, second_y, em_tensor, mu_s, mu_d, var_s, var_d = model(original_HSI.unsqueeze(0).to(device))

    en_abundance = torch.squeeze(first_a).T

    en_abundance = torch.reshape(en_abundance, [endmember_number, col * line])
    en_abundance = en_abundance.T
    en_abundance = torch.reshape(en_abundance, [col, line, endmember_number])
    abundance_GT = torch.reshape(abundance_GT, [endmember_number, col * line])
    abundance_GT = abundance_GT.T
    abundance_GT = torch.reshape(abundance_GT, [line, col, endmember_number])
    en_abundance = en_abundance.cpu().detach().numpy()
    abundance_GT = abundance_GT.cpu().detach().numpy()

    endmember_hat = em_tensor.cpu().detach().numpy()
    endmember_hat = np.squeeze(endmember_hat)
    endmember_hat = np.transpose(endmember_hat, (2, 1, 0))
    endmember_hat = np.mean(endmember_hat, axis=2)
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
