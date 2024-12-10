import os
import numpy as np
import torch.utils
import torch.utils.data
from torch import nn
import scipy.io as sio
import time
from model import DAEN
from utility import plotEndmembersAndGT, plotAbundancesSimple, plotAbundancesGT, plotAbundancesAndGT, load_HSI, hyperVca, SADloss
import pandas as pd
import random
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cosine
# train: 调用模型训练得到结果及参数等
start_time = time.time()


def cos_dist(x1, x2):
    return cosine(x1, x2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                "dc": 'DC2',}
dataset = "Urban"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
end = []
abu = []
r = []

if dataset == "Jasper":
    lambda_kl = 1
    lambda_vol = 1
    epochs = 100
if dataset == "Samson":
    lambda_kl = 1
    lambda_vol = 1
    epochs = 180
if dataset == "Urban":
    lambda_kl = 1
    lambda_vol = 1
    epochs = 130
if dataset == "dc":
    lambda_kl = 0.3
    lambda_vol = 4
    epochs = 130

num_bands = data.shape[1]
num_endmembers = hsi.gt.shape[0]
num_pixels = data.shape[0]
num_cols = hsi.abundance_gt.shape[0]
batchsize = num_pixels // 10
lr = 0.005
z_dim = 2
num_runs = 1
output_path = 'D:/毕设/Results'
method_name = 'DAEN'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


for run in range(1, num_runs + 1):
    """
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置随机种子，保证每次运行结果一致，如果多次运行一定要加在run循环下面，因为这里是有加载数据的部分。
    # 保证运行结果一致后，可以固定其它参数，逐渐调整一个参数，逐渐手动调整。
    # 有自动调参方法，如网格搜索、随机搜索、粒子群优化。毕设完成后学习一下。
    """
    train_db = torch.tensor(data)
    train_db = torch.utils.data.TensorDataset(train_db)
    train_db = torch.utils.data.DataLoader(train_db, batch_size=batchsize, shuffle=True)  # shuffle 是指是否将数据集随机化

    model = DAEN(num_endmembers, num_bands, z_dim).to(device)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    A_true = hsi.abundance_gt

    losses = []
    print('Start training!', 'run:', run)
    model.train()

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name
    abundanceAndGT_name = datasetnames[dataset] + 'withGT_run' + str(run)
    abundanceAndGT_path = abundance_folder + '/' + abundanceAndGT_name

    for epoch in range(epochs):
        for step, y in enumerate(train_db):
            y = y[0].to(device)
            y_hat, mu, log_var, a, em_tensor = model(y)

            loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0]

            loss_sad = SADloss(y, y_hat)

            kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
            kl_div = kl_div.sum() / y.shape[0]
            # KL balance of VAE
            kl_div = torch.max(kl_div, torch.tensor(0.2).to(device))

            em_bar = em_tensor.mean(dim=1, keepdim=True)
            loss_minvol = ((em_tensor - em_bar) ** 2).sum() / y.shape[0] / num_endmembers / num_bands

            loss = loss_sad + lambda_kl * kl_div + lambda_vol * loss_minvol

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        if (epoch + 1) % 10 == 0:
            print('epoch:', epoch + 1, 'loss:', losses[epoch])
    # 评估模式。而非训练模式。在评估模式下,batchNorm层,dropout层等用于优化训练而添加的网络层会被关闭,从而使得评估时不会发生偏移
    with torch.no_grad():  # 训练好模型后，使用with torch.no_grad()则不会计算梯度值，然后并不会改变模型的参数，只是看了训练的效果。精度评价环节
        model.eval()
        y_hat, mu, log_var, A, em_hat = model(torch.tensor(data).to(device))
        # 训练好模型后输入全部数据  DAEU 就是每次输入设置的 num_spectral 个数据，没有全部输入过

        em_hat = em_hat.data.cpu().numpy()  # 像元 * 端元 * 波段 VAE生成了每一个像元里的端元，考虑了端元变异性
        em_shape = em_hat.transpose(1, 0, 2)

        em_hat = np.mean(em_hat, axis=0)  # torch数组要用torch.mean  为了做实验取所有像元中提取的端元的平均值

        A_hat = A.cpu().numpy()
        A_true = A_true
        A_hat = np.reshape(A_hat, (num_cols, num_cols, num_endmembers))

        plotEndmembersAndGT(em_hat, hsi.gt, endmember_path, end)
        plotAbundancesSimple(A_hat, hsi.abundance_gt, abundance_path, abu)
        plotAbundancesAndGT(A_hat, hsi.abundance_gt, abundanceAndGT_path)
        Y_hat = y_hat.cpu().numpy()
        armse_y = np.sqrt(np.mean(np.mean((Y_hat - data) ** 2, axis=1)))  # 除以列数(波段数)求均值，即求每一行（每一个像元）的均值
        # np.mean若不指定axis是对所有元素求均值
        r.append(armse_y)

        sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'EM': em_hat,
                                                                                  'A': A_hat,
                                                                                  'Y_hat': Y_hat
                                                                                  })

    print('-' * 70)

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
