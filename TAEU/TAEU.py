import random
import numpy as np
import os
import pickle
import time
import scipy.io as sio
import torch
import torch.nn as nn
import utils
from model import AutoEncoder
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# 这里是为了确保每次运行此程序后结果一致，别的代码也不用加
"""

class Train_test:
    def __init__(self, dataset, device):
        super(Train_test, self).__init__()
        self.device = device
        self.dataset = dataset
        self.data = utils.Data(dataset, device)
        self.P, self.L, self.col = self.data.get("num_endmembers"), self.data.get("num_bands"), self.data.get(
            "num_cols")
        self.loader = self.data.get_loader(batch_size=self.col ** 2)
        self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        if dataset == 'Samson' or dataset == 'Jasper':
            self.LR, self.EPOCH = 3e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.02
            self.weight_decay_param = 0
        elif dataset == 'dc_test':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.03
            self.weight_decay_param = 1e-5
        elif dataset == 'Urban4_new':
            self.LR, self.EPOCH = 3e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 2000, 0.03
            self.weight_decay_param = 1e-4
        elif dataset == 'sim1020':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 1000, 0.03
            self.weight_decay_param = 3e-5
        elif dataset == 'berlin_test':
            self.LR, self.EPOCH = 3e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.03
            self.weight_decay_param = 0
        elif dataset == 'apex_new':
            self.LR, self.EPOCH = 9e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 4000, 0.05
            self.weight_decay_param = 4e-5
        elif dataset == 'moni30':
            self.LR, self.EPOCH = 4e-4, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 3000, 0.05
            self.weight_decay_param = 4e-5
        elif dataset == 'houston_lidar':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 2000, 0.03
            self.weight_decay_param = 4e-5
        elif dataset == 'moffett':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.03
            self.weight_decay_param = 4e-5
        elif dataset == 'muffle':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 2000, 0.03
            self.weight_decay_param = 4e-5

    def run(self, num_runs):
        end = []
        abu = []
        r = []
        time_start = time.time()

        output_path = 'D:/毕设/Results'
        method_name = 'TAEU'
        mat_folder = output_path + '/' + method_name + '/' + self.dataset + '/' + 'mat'
        endmember_folder = output_path + '/' + method_name + '/' + self.dataset + '/' + 'endmember'
        abundance_folder = output_path + '/' + method_name + '/' + self.dataset + '/' + 'abundance'
        if not os.path.exists(mat_folder):
            os.makedirs(mat_folder)
        if not os.path.exists(endmember_folder):
            os.makedirs(endmember_folder)
        if not os.path.exists(abundance_folder):
            os.makedirs(abundance_folder)

        for run in range(1, num_runs + 1):
            """
            seed = 1
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            # 这里是为了确保每次循环run后结果一致
            # 代码在GPU上运行，还需要设置torch.backends.cudnn.deterministic = True，以确保使用相同的输入和参数时，CUDA卷积运算的结果始终是确定的。
            # 其它有的代码如CyCUNet里没加这行（有卷积运算）也能保证随机性一致，不知道为啥
            # 这将会禁用一些针对性能优化的操作，因此可能会导致训练速度变慢
            """
            net = AutoEncoder(P=self.P, L=self.L, size=self.col,
                              patch=self.patch, dim=self.dim).to(self.device)

            total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"Total number of parameters: {total_params}")

            net.apply(net.weights_init)
            # 首先，net.apply(net.weights_init) 调用了 net 的 weights_init 方法来对神经网络 net 的权重进行初始化。
            # 然后，代码通过 net.state_dict() 方法获取了神经网络 net 的所有状态字典，并将其存储在 model_dict 变量中。
            # 接下来，代码修改了 model_dict 中的 decoder.0.weight 参数的值，将其设置为 self.init_weight。
            # 这里的 decoder.0.weight 是指神经网络 net 中第一层解码器的权重。
            # 最后，代码通过 net.load_state_dict(model_dict) 方法将修改后的状态字典重新加载到神经网络 net 中，使得神经网络的权重被初始化为新的值。
            model_dict = net.state_dict()
            model_dict['decoder.0.weight'] = self.init_weight
            net.load_state_dict(model_dict)

            loss_func = nn.MSELoss(reduction='mean')
            loss_func2 = utils.SAD(self.L)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
            apply_clamp_inst1 = utils.NonZeroClipper()

            endmember_name = self.dataset + '_run' + str(run)
            endmember_path = endmember_folder + '/' + endmember_name
            abundance_name = self.dataset + '_run' + str(run)
            abundance_path = abundance_folder + '/' + abundance_name
            abundanceAndGT_name = self.dataset + 'withGT_run' + str(run)

            net.train().cuda()
            print('Start training!', 'run:', run)
            for epoch in range(self.EPOCH):
                for i, x in enumerate(self.loader):
                    x = x.transpose(1, 0).view(1, -1, self.col, self.col).cuda()
                    abu_est, re_result = net(x)

                    loss_re = self.beta * loss_func(re_result, x)
                    loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                          x.view(1, self.L, -1).transpose(1, 2))
                    loss_sad = self.gamma * torch.sum(loss_sad).float()
                    ab = abu_est.view(-1, self.col * self.col)
                    # osp = utils.OSP(ab, self.P)
                    # loss3 = torch.sum(torch.pow(torch.abs(ab) + 1e-8, 0.8))
                    total_loss = loss_re + loss_sad

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    net.decoder.apply(apply_clamp_inst1)

                    if epoch % 20 == 0:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)

                scheduler.step()

            print('-' * 70)

            # Testing ================
            net.eval()
            x = self.data.get("hs_img").transpose(1, 0).view(1, -1, self.col, self.col)
            abu_est, re_result = net(x)  # abuest 1 3 95 95

            abu_est = abu_est.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
            true_endmem = self.data.get("end_mem").cpu().numpy()
            true_abundance = self.data.get("abd_map").cpu().numpy()
            est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
            est_endmem = est_endmem.reshape((self.L, self.P))
            est_endmem = est_endmem.T
            utils.plotEndmembersAndGT(est_endmem, true_endmem, endmember_path, end)
            utils.plotAbundancesSimple(abu_est, true_abundance, abundance_path, abu)
            sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'M': est_endmem,
                                                                                      'A': abu_est,
                                                                                      })

            x = x.view(-1, self.col * self.col).permute(1, 0).detach().cpu().numpy()
            re_result = re_result.view(-1, self.col * self.col).permute(1, 0).detach().cpu().numpy()
            armse_y = np.sqrt(np.mean(np.mean((re_result - x) ** 2, axis=1)))
            r.append(armse_y)

        end = np.reshape(end, (-1, self.data.get("num_endmembers") + 1))
        abu = np.reshape(abu, (-1, self.data.get("num_endmembers") + 1))
        dt = pd.DataFrame(end)
        dt2 = pd.DataFrame(abu)
        dt3 = pd.DataFrame(r)
        dt.to_csv(
            output_path + '/' + method_name + '/' + self.dataset + '/' + self.dataset + '各端元SAD及mSAD运行结果.csv')
        dt2.to_csv(
            output_path + '/' + method_name + '/' + self.dataset + '/' + self.dataset + '各丰度图RMSE及mRMSE运行结果.csv')
        dt3.to_csv(output_path + '/' + method_name + '/' + self.dataset + '/' + self.dataset + '重构误差RE运行结果.csv')
        abundanceGT_path = output_path + '/' + method_name + '/' + self.dataset + '/' + self.dataset + '参照丰度图'
        utils.plotAbundancesGT(true_abundance, abundanceGT_path)
        time_end = time.time()
        print('程序运行时间为:', time_end - time_start)


tmod = Train_test(dataset='Urban4_new', device=device)  # 要用Urban4 new 波段数必须是超参数patch(5)的倍数
tmod.run(num_runs=10)
