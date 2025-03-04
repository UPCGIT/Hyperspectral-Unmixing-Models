import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from evaluation import compute_rmse, compute_sad
from utils import print_args, SparseLoss, NonZeroClipper, MinVolumn
from data_loader import set_loader
from model import Init_Weights, MUNet
from utility import plotAbundancesGT, plotAbundancesSimple, plotEndmembersAndGT, reconstruct
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import argparse
import random
import time
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--fix_random', action='store_true', help='fix randomness')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--patch', default=10, type=int, help='input data size')
parser.add_argument('--learning_rate_en', default=3e-4, type=float, help='learning rate of encoder')
parser.add_argument('--learning_rate_de', default=1e-4, type=float, help='learning rate of decoder')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='network parameter regularization')
parser.add_argument('--lamda', default=3e-2, type=float, help='sparse regularization')
parser.add_argument('--reduction', default=2, type=int, help='squeeze reduction')
parser.add_argument('--delta', default=1, type=float, help='delta coefficient')
parser.add_argument('--gamma', default=0.8, type=float, help='learning rate decay')
parser.add_argument('--epoch', default=50, type=int, help='number of epoch')
parser.add_argument('--dataset', choices=['muffle', 'houston170'], default='muffle', help='dataset to use')
args = parser.parse_args()
num_runs = 1

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    if torch.cuda.is_available():
        print('GPU is true')
        print('cuda version: {}'.format(torch.version.cuda))
    else:
        print('CPU is true')

    if args.fix_random:
        # init seed within each thread
        """
        manualSeed = args.seed
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)"""
        # NOTE: literally you should uncomment the following, but slower
        cudnn.deterministic = True
        cudnn.benchmark = False
        print('Warning: You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')
    else:
        cudnn.benchmark = True
    print("Using GPU: {}".format(args.gpu_id))

    abund = []
    endme = []
    output_path = 'D:/study/code/Results'
    method_name = 'MUNet'
    mat_folder = output_path + '/' + method_name + '/' + args.dataset + '/' + 'mat'
    endmember_folder = output_path + '/' + method_name + '/' + args.dataset + '/' + 'endmember'
    abundance_folder = output_path + '/' + method_name + '/' + args.dataset + '/' + 'abundance'
    if not os.path.exists(mat_folder):
        os.makedirs(mat_folder)
    if not os.path.exists(endmember_folder):
        os.makedirs(endmember_folder)
    if not os.path.exists(abundance_folder):
        os.makedirs(abundance_folder)
    for run in range(1, num_runs + 1):
        print('Start training!', 'run:', run)
        # create dataset and model
        train_loaders, test_loaders, label, M_init, M_true, num_classes, band, col, row, ldr_dim = set_loader(args)
        net = MUNet(band, num_classes, ldr_dim, args.reduction).cuda()

        # initialize net parameters and endmembers
        if args.dataset == 'muffle':
            position = np.array([0, 2, 1, 3, 4])  # muffle
            Init_Weights(net, 'xavier', 1)
        elif args.dataset == 'houston170':
            position = np.array([0, 1, 2, 3])  # houston170
            Init_Weights(net, 'xavier', 1)

        net_dict = net.state_dict()
        net_dict['decoder.0.weight'] = M_init
        net.load_state_dict(net_dict)

        endmember_name = args.dataset + '_run' + str(run)
        endmember_path = endmember_folder + '/' + endmember_name

        abundance_name = args.dataset + '_run' + str(run)
        abundance_path = abundance_folder + '/' + abundance_name

        # loss funtion and regularization
        apply_nonegative = NonZeroClipper()
        loss_func = nn.MSELoss()
        criterionSparse = SparseLoss(args.lamda)
        criterionVolumn = MinVolumn(band, num_classes, args.delta)

        # optimizer setting
        params = map(id, net.decoder.parameters())
        ignored_params = list(set(params))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        optimizer = torch.optim.Adam(
            [{'params': base_params}, {'params': net.decoder.parameters(), 'lr': args.learning_rate_de}],
            lr=args.learning_rate_en, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.gamma)  # gammma:学习率下降

        time_start = time.time()
        for epoch in range(args.epoch):
            for i, traindata in enumerate(train_loaders):
                net.train()

                x, y = traindata
                x = x.cuda()
                y = y.cuda()

                abu, output = net(x, y)
                output = torch.reshape(output, (output.shape[0], band))
                x = torch.reshape(x, (output.shape[0], band))

                # reconstruction loss
                MSE_loss = torch.mean(torch.acos(torch.sum(x * output, dim=1) /
                                                 (torch.norm(output, dim=1, p=2) * torch.norm(x, dim=1, p=2))))
                # sparsity and minimum volume regularization
                MSE_loss += criterionSparse(abu) + criterionVolumn(net.decoder[0].weight)

                optimizer.zero_grad()
                MSE_loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                optimizer.step()
                net.decoder.apply(apply_nonegative)

            if epoch % 1 == 0:
                print(
                    'Epoch: {:d} | Train Unmix Loss: {:.5f} | RE Loss: {:.5f} | Sparsity Loss: {:.5f} | Minvol: {:.5f}'
                    .format(epoch, MSE_loss, loss_func(output, x), criterionSparse(abu),
                            criterionVolumn(net.decoder[0].weight)))
                net.eval()
                for k, testdata in enumerate(test_loaders):
                    x, y = testdata
                    x = x.cuda()
                    y = y.cuda()

                    abu_est, output = net(x, y)

                abu_est = torch.reshape(abu_est.squeeze(-1).permute(2, 1, 0), (num_classes, row, col)).permute(0, 2,
                                                                                                               1).cpu().data.numpy()
                edm_result = torch.reshape(net.decoder[0].weight, (band, num_classes)).cpu().data.numpy()
                print('RMSE: {:.5f} | SAD: {:.5f}'.format(compute_rmse(abu_est[position, :, :], label),
                                                          compute_sad(M_true, edm_result[:, position])))
                print('**********************************')

            scheduler.step()
        time_end = time.time()
        # model evaluation
        net.eval()
        print(net.spectral_se)
        for i, testdata in enumerate(test_loaders):
            x, y = testdata
            x = x.cuda()
            y = y.cuda()

            abu, output = net(x, y)

        # compute metric
        abu_est = torch.reshape(abu.squeeze(-1).permute(2, 1, 0), (num_classes, row, col)).permute(0, 2,
                                                                                                   1).cpu().data.numpy()
        edm_result = torch.reshape(net.decoder[0].weight, (band, num_classes)).cpu().data.numpy()
        abu_est = abu_est[position, :, :]
        edm_result = edm_result[:, position]

        RMSE = compute_rmse(label, abu_est)
        SAD = compute_sad(M_true, edm_result)

        plotAbundancesSimple(abu_est.transpose(1, 2, 0), label.transpose(2, 1, 0), abundance_path, abund)
        plotEndmembersAndGT(edm_result.T, M_true.T, endmember_path, endme)

        print('**********************************')
        print('RMSE: {:.5f} | SAD: {:.5f}'.format(RMSE, SAD))
        print('**********************************')
        print('total computational cost:', time_end - time_start)
        print('**********************************')

        # print hyperparameter setting and save result
        print_args(vars(args))
        save_path = str(args.dataset) + '_result.mat'
        sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'A': abu_est.transpose(1, 2, 0),
                                                                                  'E': edm_result.T, })

        print('-' * 70)

    end_time = time.time()
    end = np.reshape(endme, (-1, M_true.shape[1] + 1))
    abu = np.reshape(abund, (-1, M_true.shape[1] + 1))
    dt = pd.DataFrame(end)
    dt2 = pd.DataFrame(abu)
    dt.to_csv(output_path + '/' + method_name + '/' + args.dataset + '/' + args.dataset + '各端元SAD及mSAD运行结果.csv')
    dt2.to_csv(
        output_path + '/' + method_name + '/' + args.dataset + '/' + args.dataset + '各丰度图RMSE及mRMSE运行结果.csv')
    abundanceGT_path = output_path + '/' + method_name + '/' + args.dataset + '/' + args.dataset + '参照丰度图'
    plotAbundancesGT(label.transpose(1, 2, 0), abundanceGT_path)
