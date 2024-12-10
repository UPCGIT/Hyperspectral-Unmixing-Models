import numpy as np
import matplotlib.pyplot as plt
import torch.optim
from utility import get_noise, get_params, Endmember_extract, OSP
from utility import plotEndmembersAndGT, plotAbundancesSimple, plotAbundancesGT, load_HSI
from model import MiSiCNet
import scipy.linalg
import time
import os
import random
import scipy.io as sio
import pandas as pd

start_time = time.time()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
PLOT = False

datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson'}
dataset = "Samson" # 下面有参数要调整
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
end = []
abu = []
r = []
save = []

img_np_gt = hsi.image
img_np_gt = img_np_gt.transpose(2, 0, 1)
[num_bands, num_cols, num_cols] = img_np_gt.shape
num_endmembers = hsi.gt.shape[0]

runs = 1
output_path = 'D:/毕设/Results'
method_name = 'MiSiCNet'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

for run in range(runs):

    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    img_resh = np.reshape(img_np_gt, (num_bands, num_cols * num_cols))
    V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
    PC = np.diag(SS) @ U
    img_resh_DN = V[:, :num_endmembers] @ V[:, :num_endmembers].transpose(1, 0) @ img_resh
    img_resh_np_clip = np.clip(img_resh_DN, 0, 1)
    II, III = Endmember_extract(img_resh_np_clip, num_endmembers)
    E_np1 = img_resh_np_clip[:, II]
    # Set up Simulated
    INPUT = 'noise'  # 'meshgrid'  # get_noisy
    pad = 'reflection'  # MiSiCNet
    need_bias = True  # MiSiCNet
    OPT_OVER = 'net'  # get_params

    LR1 = 0.001 # Jasper 和 Samson 0.001  Urban 0.01 动态调整
    exp_weight = 0.99
    losses = []

    epochs = 600 # Jasper 和 Samson 600  Urban 400 动态调整
    input_depth = img_np_gt.shape[0]
    E_torch = torch.from_numpy(E_np1).type(dtype)

    net = MiSiCNet(input_depth, need_bias, pad, num_endmembers, num_bands, num_cols, num_cols)
    net.cuda()
    net.dconv[0].weight = torch.nn.Parameter(E_torch.view(num_bands, num_endmembers))
    # 这个权重初始化一开始是放在每次epoch的train函数里面，这样的话每次epoch都会重新初始化权重，导致无法更新参数

    img_noisy_torch = torch.from_numpy(img_resh_DN).view(1, num_bands, num_cols, num_cols).type(dtype)
    net_input1 = get_noise(input_depth, INPUT,
                           (img_np_gt.shape[1], img_np_gt.shape[2])).type(dtype).detach()

    out_avg = True

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name


    def my_loss(target, End2, lamb, out_, l1):
        loss1 = 0.5 * torch.norm((out_.transpose(1, 0).view(1, num_bands, num_cols, num_cols) - target), 'fro') ** 2
        O = torch.mean(target.view(num_bands, num_cols * num_cols), 1).type(dtype).view(num_bands, 1)
        B = torch.from_numpy(np.identity(num_endmembers)).type(dtype)
        loss2 = torch.norm(torch.mm(End2, B.view((num_endmembers, num_endmembers))) - O, 'fro') ** 2
        loss3 = torch.sum(torch.pow(torch.abs(l1) + 1e-8, 0.8))
        loss4 = OSP(l1, num_endmembers)
        return loss1 + lamb * loss2 + loss3


    i = 0


    def train():
        global i, out_LR_np, out_avg, out_avg_np, out_spec
        # 定义全局变量 使得可以在训练后在其它处调用 其实效果应该和return值一样

        out_LR, out_spec = net(net_input1)
        # Smoothing
        if out_avg is None:
            out_avg = out_LR.detach()
        else:
            out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)

        total_loss = my_loss(img_noisy_torch, net.dconv[0].weight, 100, out_spec, out_avg)
        total_loss.backward()

        if PLOT and i % 20 == 0:
            out_LR_np = out_LR.detach().cpu().squeeze().numpy()
            out_avg_np = out_avg.detach().cpu().squeeze().numpy()
            out_LR_np = np.clip(out_LR_np, 0, 1)
            out_avg_np = np.clip(out_avg_np, 0, 1)
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
            ax1.imshow(np.stack((out_LR_np[2, :, :], out_LR_np[1, :, :], out_LR_np[0, :, :]), 2))  # 显示左列
            ax2.imshow(np.stack((out_avg_np[2, :, :], out_avg_np[1, :, :], out_avg_np[0, :, :]), 2))  # 显示右列
            plt.show()

        i += 1
        losses.append(total_loss.detach().cpu().numpy())
        if (epoch + 1) % 100 == 0:
            print('epoch:', epoch + 1, 'loss:', losses[epoch])
        return total_loss


    p11 = get_params(OPT_OVER, net, net_input1)
    optimizer = torch.optim.Adam(p11, lr=LR1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    for epoch in range(epochs):
        optimizer.zero_grad()
        train()  # 一次epoch跑一次 i+1
        optimizer.step()
        net.dconv[0].weight.data[net.dconv[0].weight <= 0] = 0
        net.dconv[0].weight.data[net.dconv[0].weight >= 1] = 1

    out_spec = out_spec.detach().cpu().squeeze().numpy()
    Eest = net.dconv[0].weight.detach().cpu().squeeze().numpy()
    out_avg_np = out_avg.detach().cpu().squeeze().numpy()
    out_avg_np = out_avg_np.transpose(1, 2, 0)
    plotEndmembersAndGT(Eest.T, hsi.gt, endmember_path, end, save)
    plotAbundancesSimple(out_avg_np, hsi.abundance_gt, abundance_path, abu)
    armse_y = np.sqrt(np.mean(np.mean((out_spec - data) ** 2, axis=1)))
    r.append(armse_y)

    sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'EM': Eest,
                                                                              'A': out_avg_np,
                                                                              'Y_hat': out_spec
                                                                              })
    print('-' * 70)

save = np.reshape(save, (-1, num_endmembers, num_bands))
fig = plt.figure(num=1, figsize=(9, 9))
n = int(num_endmembers // 2)
if num_endmembers % 2 != 0:
    n = n + 1
for i in range(runs):
    for j in range(num_endmembers):
        ax = plt.subplot(2, n, j + 1)
        plt.plot(save[i, j, :], 'b', linewidth=1.0)
        plt.plot(hsi.gt[j, :], 'r', linewidth=1.0)
        ax.get_xaxis().set_visible(False)
plt.tight_layout()
plt.savefig(output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[dataset] +
            '多次运行端元对照图.png')

end = np.reshape(end, (-1, num_endmembers + 1))
abu = np.reshape(abu, (-1, num_endmembers + 1))
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
end_time = time.time()
print('程序运行时间为:', end_time - start_time, 's')
