import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch.utils.data
import scipy.io as sio
import torchvision.transforms as transforms
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

matplotlib.rc("font", family='Microsoft YaHei')


def OSP(B, R):
    dots = 0.0
    B = torch.squeeze(B)
    B = B.view(R, -1)
    for i in range(R):
        for j in range(i + 1, R):
            A1 = B[i, :]
            A2 = B[j, :]
            dot = torch.sum(A1*A2)
            dots = dots + dot
    return dots


def pca(X, d):
    N = np.shape(X)[1]
    xMean = np.mean(X, axis=1, keepdims=True)
    XZeroMean = X - xMean
    [U, S, V] = np.linalg.svd((XZeroMean @ XZeroMean.T) / N)
    Ud = U[:, 0:d]
    return Ud


def hyperVca(M, q):
    '''
    M : [L,N]
    '''
    L, N = np.shape(M)

    rMean = np.mean(M, axis=1, keepdims=True)
    RZeroMean = M - rMean
    U, S, V = np.linalg.svd(RZeroMean @ RZeroMean.T / N)
    Ud = U[:, 0:q]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(M ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (q / L) * P_R) / (P_R - P_Rp)))
    snrEstimate = SNR
    # print('SNR estimate [dB]: %.4f' % SNR[0, 0])
    # Determine which projection to use.
    SNRth = 18 + 10 * np.log(q)

    if SNR > SNRth:
        d = q
        # [Ud, Sd, Vd] = svds((M * M.')/N, d);
        U, S, V = np.linalg.svd(M @ M.T / N)
        Ud = U[:, 0:d]
        Xd = Ud.T @ M
        u = np.mean(Xd, axis=1, keepdims=True)
        # print(Xd.shape, u.shape, N, d)
        Y = Xd / np.sum(Xd * u, axis=0, keepdims=True)

    else:
        d = q - 1
        r_bar = np.mean(M.T, axis=0, keepdims=True).T
        Ud = pca(M, d)

        R_zeroMean = M - r_bar
        Xd = Ud.T @ R_zeroMean
        # Preallocate memory for speed.
        # c = np.zeros([N, 1])
        # for j in range(N):
        #     c[j] = np.linalg.norm(Xd[:, j], ord=2)
        c = [np.linalg.norm(Xd[:, j], ord=2) for j in range(N)]
        # print(type(c))
        c = np.array(c)
        c = np.max(c, axis=0, keepdims=True) @ np.ones([1, N])
        Y = np.concatenate([Xd, c.reshape(1, -1)])
    e_u = np.zeros([q, 1])
    # print('*',e_u)
    e_u[q - 1, 0] = 1
    A = np.zeros([q, q])
    # idg - Doesntmatch.
    # print (A[:, 0].shape)
    A[:, 0] = e_u[0]
    I = np.eye(q)
    k = np.zeros([N, 1])

    indicies = np.zeros([q, 1])
    for i in range(q):  # i=1:q
        w = np.random.random([q, 1])

        # idg - Oppurtunity for speed up here.
        tmpNumerator = (I - A @ np.linalg.pinv(A)) @ w
        # f = ((I - A * pinv(A)) * w) / (norm(tmpNumerator));
        f = tmpNumerator / np.linalg.norm(tmpNumerator)

        v = f.T @ Y
        k = np.abs(v)

        k = np.argmax(k)
        A[:, i] = Y[:, k]
        indicies[i] = k

    indicies = indicies.astype('int')
    # print(indicies.T)
    if (SNR > SNRth):
        U = Ud @ Xd[:, indicies.T[0]]
    else:
        U = Ud @ Xd[:, indicies.T[0]] + r_bar

    return U


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, transform=None):
        self.img = img.float()
        self.transform = transform

    def __getitem__(self, index):
        img = self.img[index]
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img)


class HSI:
    def __init__(self, data, rows, cols, gt, abundance_gt):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows
        self.image = np.reshape(data, (self.rows, self.cols, self.bands))
        self.gt = gt
        self.abundance_gt = abundance_gt

    def array(self):
        """返回 像元*波段 的数据阵列（array)"""

        return np.reshape(self.image, (self.rows * self.cols, self.bands))


def load_HSI(path):
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')

    numpy_array = np.asarray(data['Y'], dtype=np.float32)  # Y是波段*像元
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()

    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None

    if 'S_GT' in data.keys():
        abundance_gt = np.asarray(data['S_GT'], dtype=np.float32)
    else:
        abundance_gt = None

    return HSI(numpy_array, n_rows, n_cols, gt, abundance_gt)


class Data:
    def __init__(self, dataset, device):
        super(Data, self).__init__()

        data_path = "D:/毕设/Datasets/" + dataset + ".mat"
        hsi = load_HSI(data_path)
        blocks = sio.loadmat(r'D:\毕设\code\MsCM-Net(10.23)\blocks_samson.mat')['samson']

        self.P = hsi.gt.shape[0]
        self.L = hsi.gt.shape[1]
        self.col = hsi.cols
        self.A = hsi.abundance_gt
        self.Y = hsi.array()
        self.M = hsi.gt
        end_vca = hyperVca(self.Y.T, self.P)

        self.Y = torch.from_numpy(self.Y).to(device)
        self.A = torch.from_numpy(self.A).to(device)
        self.M = torch.from_numpy(self.M)
        self.M1 = torch.tensor(end_vca).to(device)

    def get(self, typ):
        if typ == "hs_img":
            return self.Y
        elif typ == "abd_map":
            return self.A
        elif typ == "end_mem":
            return self.M
        elif typ == "init_weight":
            return self.M1
        elif typ == "num_endmembers":
            return self.P
        elif typ == "num_bands":
            return self.L
        elif typ == "num_cols":
            return self.col

    def get_loader(self, batch_size):
        train_dataset = TrainData(img=self.Y, transform=transforms.Compose([]))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        return train_loader


def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad


def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    inp = torch.reshape(inputs, (band, h * w))
    out = torch.norm(inp, p='nuc')
    return out


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, inp, decay):
        inp = torch.sum(inp, 0, keepdim=True)
        loss = Nuclear_norm(inp)
        return decay * loss


class SumToOneLoss(nn.Module):
    def __init__(self, device):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float, device=device))
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, inp):
        target_tensor = self.one
        return target_tensor.expand_as(inp)

    def __call__(self, inp, gamma_reg):
        inp = torch.sum(inp, 1)
        target_tensor = self.get_target_tensor(inp)
        loss = self.loss(inp, target_tensor)
        return gamma_reg * loss


class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        try:
            input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                              inp.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                               target.view(-1, self.num_bands, 1)))

            summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation / (input_norm * target_norm))

        except ValueError:
            return 0.0

        return angle


class SID(nn.Module):
    def __init__(self, epsilon: float = 1e5):
        super(SID, self).__init__()
        self.eps = epsilon

    def forward(self, inp, target):
        normalize_inp = (inp / torch.sum(inp, dim=0)) + self.eps
        normalize_tar = (target / torch.sum(target, dim=0)) + self.eps
        sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) +
                        normalize_tar * torch.log(normalize_tar / normalize_inp))

        return sid


def numpy_MSE(y_true, y_pred):  # 错写成MSE了，实际上是算RMSE，将错就错了
    num_cols = y_pred.shape[0]
    num_rows = y_pred.shape[1]
    diff = y_true - y_pred
    squared_diff = np.square(diff)
    mse = squared_diff.sum() / (num_rows * num_cols)
    rmse = np.sqrt(mse)
    return rmse


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_MSE(a[:, :, i], agt[:, :, j])

        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance


def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos > 1.0: cos = 1.0
    return np.arccos(cos)


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    SAD_matrix = np.zeros((num_endmembers, num_endmembers))
    SAD_index = np.zeros(num_endmembers).astype(int)
    SAD_endmember = np.zeros(num_endmembers)
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    e = endmembers.copy()
    egt = endmembersGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            SAD_matrix[i, j] = numpy_SAD(e[i, :], egt[j, :])

        SAD_index[i] = np.nanargmin(SAD_matrix[i, :])
        SAD_endmember[i] = np.nanmin(SAD_matrix[i, :])
        egt[SAD_index[i], :] = np.inf
    return SAD_index, SAD_endmember


def plotEndmembersAndGT(endmembers, endmembersGT, endmember_path, sadsave):
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1
    SAD_index, SAD_endmember = order_endmembers(endmembersGT, endmembers)
    fig = plt.figure(num=1, figsize=(9, 9))
    plt.clf()
    title = "mSAD: " + np.array2string(SAD_endmember.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x}) + " radians"
    plt.rcParams.update({'font.size': 15})
    st = plt.suptitle(title)

    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[SAD_index[i], :], 'r', linewidth=1.0)
        plt.plot(endmembersGT[i, :], 'k', linewidth=1.0)
        ax.set_title(format(numpy_SAD(endmembers[SAD_index[i], :], endmembersGT[i, :]), '.3f'))
        ax.get_xaxis().set_visible(False)
        sadsave.append(numpy_SAD(endmembers[SAD_index[i], :], endmembersGT[i, :]))

    sadsave.append(SAD_endmember.mean())
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.86)
    plt.savefig(endmember_path + '.png')
    """plt.draw()
    plt.pause(0.1)
    plt.close()"""


def order_abundance(abundance, abundanceGT):
    num_endmembers = abundance.shape[2]
    abundance_matrix = np.zeros((num_endmembers, num_endmembers))
    abundance_index = np.zeros(num_endmembers).astype(int)
    MSE_abundance = np.zeros(num_endmembers)
    a = abundance.copy()
    agt = abundanceGT.copy()
    for i in range(0, num_endmembers):
        for j in range(0, num_endmembers):
            abundance_matrix[i, j] = numpy_MSE(a[:, :, i], agt[:, :, j])

        abundance_index[i] = np.nanargmin(abundance_matrix[i, :])
        MSE_abundance[i] = np.nanmin(abundance_matrix[i, :])
        agt[:, :, abundance_index[i]] = np.inf
    return abundance_index, MSE_abundance


def plotAbundancesSimple(abundances, abundanceGT, abundance_path, rmsesave):
    abundances = np.transpose(abundances, axes=[1, 0, 2])  # 把行列颠倒，第三维不动，因为方法代码里写的得到的丰度是列*行，但是如果行列数相同，倒也不影响
    num_endmembers = abundances.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    abundance_index, MSE_abundance = order_abundance(abundanceGT, abundances)
    title = "RMSE: " + np.array2string(MSE_abundance.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x})
    # cmap = 'jet'
    cmap = 'viridis'
    plt.figure(figsize=[10, 10])
    AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, abundance_index[i]], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.set_title(format(numpy_MSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i]), '.3f'))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        rmsesave.append(numpy_MSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i]))

    rmsesave.append(MSE_abundance.mean())
    plt.tight_layout()  # 用于自动调整子图参数，以便使所有子图适合整个图像区域，并尽可能地减少子图之间的重叠
    plt.rcParams.update({'font.size': 15})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    """plt.draw()
    plt.pause(0.1)
    plt.close()"""


def plotAbundancesGT(abundanceGT, abundance_path):
    num_endmembers = abundanceGT.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    title = '参照丰度图'
    cmap = 'jet'
    plt.figure(figsize=[10, 10])
    AA = np.sum(abundanceGT, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundanceGT[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()  # 用于自动调整子图参数，以便使所有子图适合整个图像区域，并尽可能地减少子图之间的重叠
    plt.rcParams.update({'font.size': 19})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    plt.draw()
    plt.pause(0.1)
    plt.close()


def plotAbundancesAndGT(abundances, abundanceGT, abundance_path):
    abundances = np.transpose(abundances, axes=[1, 0, 2])  # 把行列颠倒，第三维不动，因为方法代码里写的得到的丰度是列*行，但是如果行列数相同，倒也不影响
    num_endmembers = abundances.shape[2]
    n = num_endmembers
    if num_endmembers % 2 != 0: n = n + 1
    abundance_index, MSE_abundance = order_abundance(abundanceGT, abundances)
    title = "RMSE: " + np.array2string(MSE_abundance.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x})
    cmap = 'viridis'
    plt.figure(figsize=[10, 7])
    AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, abundance_index[i]], cmap=cmap)
        cbar = plt.colorbar(im, cax=cax, ticks=[0.2, 0.4, 0.6, 0.8], orientation='horizontal')
        cbar.ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8'])
        ax.set_title(format(numpy_MSE(abundances[:, :, abundance_index[i]], abundanceGT[:, :, i]), '.3f'))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, n + i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundanceGT[:, :, i], cmap=cmap)
        cbar = plt.colorbar(im, cax=cax, ticks=[0.2, 0.4, 0.6, 0.8], orientation='horizontal')
        cbar.ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8'])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.rcParams.update({'font.size': 15})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')
    plt.draw()
    plt.pause(0.1)
    plt.close()