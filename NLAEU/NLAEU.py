import scipy.io as sio
import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import Variable
from model import NLAEU
from utility import pretrain_dec_nonlipart, load_HSI, hyperVca, plotEndmembersAndGT, plotAbundancesSimple, plotAbundancesGT
import pandas as pd

start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'synthetic': 'synthetic5',
                'dc': 'DC2',
                'sim': 'sim1020',
                'berlin': 'berlin_test'}
dataset = "berlin"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
end = []
abu = []
r = []

EPOCH = 80 # 400
BATCH_SIZE = 2048
learning_rate = 2e-3
num_endmembers = hsi.gt.shape[0]
num_bands = data.shape[1]
num_cols = hsi.abundance_gt.shape[0]
num_runs = 10

output_path = 'D:/毕设/Results'
method_name = 'NLAEU'
mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
if not os.path.exists(mat_folder):
    os.makedirs(mat_folder)
if not os.path.exists(endmember_folder):
    os.makedirs(endmember_folder)
if not os.path.exists(abundance_folder):
    os.makedirs(abundance_folder)

# ---------------------------------------------------------------------------
for run in range(1, num_runs + 1):
    print('Start training!', 'run:', run)
    losses = []
    # --------------------- initialize the network ------------------------------
    model = NLAEU(num_endmembers, num_bands)
    EM, _, _ = hyperVca(data.T, num_endmembers)
    EM = torch.from_numpy(EM)

    model.decoder_linearpart[0].weight.data = EM
    dec_nonlipart = pretrain_dec_nonlipart(data)
    model.decoder_nonlinearpart.load_state_dict(dec_nonlipart)

    model = model.float().cuda()
    # model = model.cuda()
    MSE = MSELoss()

    params1 = map(id, model.decoder_linearpart.parameters())
    params2 = map(id, model.decoder_nonlinearpart.parameters())
    ignored_params = list(set(params1).union(set(params2)))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = Adam([
        {'params': base_params},
        {'params': model.decoder_linearpart.parameters(), 'lr': 1e-5},
        {'params': model.decoder_nonlinearpart.parameters(), 'lr': 1e-5},
    ], lr=learning_rate, weight_decay=1e-5)
    """这段代码定义了一个Adam优化器，并将模型参数分成三组：decoder_linearpart模块的参数、decoder_nonlinearpart
    模块的参数和其余模块的参数。对于前两组参数，学习率被设置为1e - 5，而对于其余模块的参数，使用默认的学习率。同时，忽略了所有
    decoder_linearpart和decoder_nonlinearpart模块的参数，这些参数将不会被优化器更新。最后，该优化器的学习率设置为
    `learning_rate`，权重衰减设置为1e - 5。该代码的目的是为了在训练神经网络时，对不同的参数组使用不同的学习率以提高训练效果。"""

    code_onehot = torch.eye(num_endmembers)
    code_onehot = Variable(code_onehot).cuda()

    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    endmember_name = datasetnames[dataset] + '_run' + str(run)
    endmember_path = endmember_folder + '/' + endmember_name
    abundance_name = datasetnames[dataset] + '_run' + str(run)
    abundance_path = abundance_folder + '/' + abundance_name

    for epoch in range(1, EPOCH + 1):
        for y in data_loader:
            pixel = y
            pixel = Variable(pixel).cuda()
            model.train().cuda()
            # ===================forward=====================

            output, vector = model(pixel)
            loss_reconstruction = MSE(output, pixel)
            loss = loss_reconstruction
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
        if (epoch + 1) % 20 == 0:
            print('epoch:', epoch + 1, 'loss:', losses[epoch])
    with torch.no_grad():
        model.eval()
        Y_hat, abundance = model(torch.tensor(data).to(device))
        abundance = abundance.cpu().numpy()
        abundance = abundance.reshape(num_cols, num_cols, num_endmembers)

        endmember = model.get_endmember(code_onehot)
        endmember = endmember.cpu().data
        endmember = endmember.numpy()

        plotEndmembersAndGT(endmember, hsi.gt, endmember_path, end)
        plotAbundancesSimple(abundance, hsi.abundance_gt, abundance_path, abu)
        Y_hat = Y_hat.cpu().numpy()
        armse_y = np.sqrt(np.mean(np.mean((Y_hat - data) ** 2, axis=1)))
        # np.mean若不指定axis是对所有元素求均值
        r.append(armse_y)

        sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'EM': endmember,
                                                                                  'A': abundance,
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
