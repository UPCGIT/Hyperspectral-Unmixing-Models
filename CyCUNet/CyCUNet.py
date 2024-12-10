import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utility import SparseKLloss, NonZeroClipper, SumToOneLoss, load_HSI, hyperVca, MyTrainData
from utility import plotEndmembersAndGT, plotAbundancesSimple, plotAbundancesGT, SAD
import numpy as np
import os
import time
import random
import pandas as pd
import scipy.io as sio
from model import AutoEncoder
from skimage.filters import threshold_otsu
time_start = time.time()
newloss = True  # 是否采用改进版本  目前跑的实验用的是True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'dc': 'DC2',
                'sim': 'sim1020',
                'berlin': 'berlin_test',
                'apex': 'apex_new',
                'moni': 'moni20',
                'houston': 'houston_test',
                'moffett': 'moffett'}
dataset = "moffett"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
end = []
abu = []
r = []

num_bands = data.shape[1]
num_endmembers = hsi.gt.shape[0]
num_pixels = data.shape[0]
num_cols = hsi.abundance_gt.shape[0]

num_runs = 1
output_path = 'D:/毕设/Results'
method_name = 'CyCUNet'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

if dataset == 'Samson':
    LR, EPOCH, batch_size = 2e-3, 480, 1
    beta, delta, gamma = 0.5, 1e-3, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-3
if dataset == 'Jasper':
    LR, EPOCH, batch_size = 8e-3, 420, 1
    beta, delta, gamma = 0.5, 1e-2, 1e-7
    sparse_decay, weight_decay_param = 0, 0
if dataset == 'Urban':
    LR, EPOCH, batch_size = 0.004, 300, 1
    beta, delta, gamma = 0.5, 0.001, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-4
if dataset == 'dc':
    LR, EPOCH, batch_size = 0.001, 300, 1
    beta, delta, gamma = 0.5, 0.001, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-5
if dataset == 'sim':
    LR, EPOCH, batch_size = 0.001, 300, 1
    beta, delta, gamma = 0.5, 0.001, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-3
if dataset == 'berlin':
    LR, EPOCH, batch_size = 2e-3, 480, 1
    beta, delta, gamma = 0.5, 1e-3, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-3
if dataset == 'apex':
    LR, EPOCH, batch_size = 0.002, 300, 1
    beta, delta, gamma = 0.5, 0.001, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-5
if dataset == 'moni':
    LR, EPOCH, batch_size = 0.001, 300, 1
    beta, delta, gamma = 0.5, 0.001, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-5
if dataset == 'houston':
    LR, EPOCH, batch_size = 0.004, 300, 1
    beta, delta, gamma = 0.5, 0.001, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-5
if dataset == 'moffett':
    LR, EPOCH, batch_size = 0.003, 500, 1
    beta, delta, gamma = 0.5, 0.001, 1e-7
    sparse_decay, weight_decay_param = 0, 1e-4


def weights_init(m):
    nn.init.kaiming_normal_(net.encoder[0].weight.data)
    nn.init.kaiming_normal_(net.encoder[4].weight.data)
    nn.init.kaiming_normal_(net.encoder[7].weight.data)


def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


def segmented_mapping(similarity_matrix, threshold, ranges):
    # Mask for values less than the threshold
    below_threshold_mask = similarity_matrix < threshold
    above_threshold_mask = similarity_matrix >= threshold

    # Calculate percentiles for values below and above the threshold
    percentiles_below = np.percentile(similarity_matrix[below_threshold_mask], [20, 40, 60, 80])
    percentiles_above = np.percentile(similarity_matrix[above_threshold_mask], [20, 40, 60, 80])

    mapped_array = np.empty_like(similarity_matrix)

    # Map values below the threshold
    for i in range(5):
        lower = percentiles_below[i - 1] if i > 0 else similarity_matrix[below_threshold_mask].min()
        upper = percentiles_below[i] if i < 4 else similarity_matrix[below_threshold_mask].max()
        mask = (similarity_matrix >= lower) & (similarity_matrix <= upper) & below_threshold_mask
        mapped_array[mask] = ranges[i][0] + (similarity_matrix[mask] - lower) / (upper - lower) * (
                ranges[i][1] - ranges[i][0])

    # Map values above the threshold
    for i in range(5):
        lower = percentiles_above[i - 1] if i > 0 else similarity_matrix[above_threshold_mask].min()
        upper = percentiles_above[i] if i < 4 else similarity_matrix[above_threshold_mask].max()
        mask = (similarity_matrix >= lower) & (similarity_matrix <= upper) & above_threshold_mask
        mapped_array[mask] = ranges[i + 5][0] + (similarity_matrix[mask] - lower) / (upper - lower) * (
                ranges[i + 5][1] - ranges[i + 5][0])

    return mapped_array


similarity_matrix = sio.loadmat(r'D:\study\code\similarity_dc.mat')['dc']
similarity_matrix = similarity_matrix[0]
thre = threshold_otsu(similarity_matrix)

# 将 similarity_matrix 视为一维数组
flattened_matrix = similarity_matrix.flatten()
q10, q20, q30, q40, q50, q60, q70, q80, q90 = np.percentile(flattened_matrix, (10, 20, 30, 40, 50, 60, 70, 80, 90))
mapping_ranges = [(0.01, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1), (1, 1.2), (1.2, 1.4), (1.4, 1.6),
                  (1.6, 1.8), (1.8, 2)]
mask = segmented_mapping(similarity_matrix, thre, mapping_ranges)
mask = torch.from_numpy(mask).unsqueeze(0).to(device)
for run in range(1, num_runs + 1):
    """
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    """
    E_VCA_init, _, _ = hyperVca(data.T, num_endmembers)
    # hyperVCA函数返回了三个值，要是只输入E_VCA_init，则会返回一个元祖类型数据，后面无法进行。可以修改函数返回值
    E_VCA_init = torch.from_numpy(E_VCA_init).unsqueeze(2).unsqueeze(3).float()
    E_VCA_init = E_VCA_init.to(device)

    load = data.T
    load = np.reshape(load, [num_bands, num_cols, num_cols])
    train_dataset = MyTrainData(img=load, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    net = AutoEncoder(num_endmembers, num_bands).cuda()
    net.apply(weights_init)
    criterionSumToOne = SumToOneLoss(gamma).cuda()
    criterionSparse = SparseKLloss(sparse_decay).cuda()

    model_dict = net.state_dict()
    model_dict['decoder1.0.weight'] = E_VCA_init
    model_dict['decoder2.0.weight'] = E_VCA_init
    net.load_state_dict(model_dict)

    loss_func = nn.MSELoss(size_average=True, reduce=True, reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    apply_clamp_inst1 = NonZeroClipper()

    print('Start training!', 'run:', run)
    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name

    for epoch in range(EPOCH):
        for i, x in enumerate(train_loader):
            net.train().cuda()
            x = x.cuda()
            abu_est1, re_result1, abu_est2, re_result2, end1 = net(x)

            loss_sumtoone = criterionSumToOne(abu_est1) + criterionSumToOne(abu_est2)
            loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)
            loss_re = beta * loss_func(re_result1, x) + (1 - beta) * loss_func(x, re_result2)
            loss_abu = delta * loss_func(abu_est1, abu_est2)
            loss_sad = 0.5 * reconstruction_SADloss(x, re_result1) + 0.5 * reconstruction_SADloss(x, re_result2)
            loss3 = torch.sum(torch.pow(torch.abs(abu_est1) + 1e-8, 0.5)) + torch.sum(torch.pow(torch.abs(abu_est2) + 1e-8, 0.5))
            total_loss = loss_abu + loss_sad + 1e-4*loss3

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
            optimizer.step()

            net.decoder1.apply(apply_clamp_inst1)
            net.decoder2.apply(apply_clamp_inst1)

        if epoch % 100 == 0:
            print('Epoch:', epoch, '| train loss: %.4f' % total_loss.cpu().data.numpy(),
                  '| abu loss: %.4f' % loss_abu.cpu().data.numpy(),
                  '| sumtoone loss: %.4f' % loss_sumtoone.cpu().data.numpy(),
                  '| re loss: %.4f' % loss_re.cpu().data.numpy())

        scheduler.step()

    with torch.no_grad():
        net.eval()
        abu_est1, re_result1, abu_est2, re_result2, end1 = net(x)
        endmember_hat = net.state_dict()["decoder1.0.weight"].cpu().numpy()
        endmember_hat = np.squeeze(endmember_hat)
        endmember_hat = endmember_hat.T

        abu_est1 = abu_est1
        abu_est1 = abu_est1.cpu().numpy()
        abu_est1 = np.squeeze(abu_est1)
        abu_est1 = np.reshape(abu_est1, [num_endmembers, num_cols * num_cols])
        abu_est1 = abu_est1.T
        abu_est1 = np.reshape(abu_est1, [num_cols, num_cols, num_endmembers])

        re_result1 = re_result1.cpu().numpy()
        re_result1 = np.squeeze(re_result1)
        re_result1 = np.reshape(re_result1, [num_bands, num_cols * num_cols])
        re_result1 = re_result1.T

        plotEndmembersAndGT(endmember_hat, hsi.gt, endmember_path, end)
        plotAbundancesSimple(abu_est1, hsi.abundance_gt, abundance_path, abu)
        armse_y = np.sqrt(np.mean(np.mean((re_result1 - data) ** 2, axis=1)))  # 除以列数(波段数)求均值，即求每一行（每一个像元）的均值
        # np.mean若不指定axis是对所有元素求均值
        r.append(armse_y)

        sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'EM': endmember_hat,
                                                                                  'A': abu_est1,
                                                                                  'Y_hat': re_result1
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
time_end = time.time()
print('程序运行时间为:', time_end - time_start)
