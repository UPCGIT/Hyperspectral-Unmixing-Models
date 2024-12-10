import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import scipy.io as sio
import h5py
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


def fill_noise(x, noise_type):
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='n', var=1. / 10):
    """
    特定的初始化方式
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = torch.from_numpy(meshgrid)
    else:
        assert False

    return net_input


def get_params(opt_over, net, net_input, downsampler=None):
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def Eucli_dist(x, y):
    a = np.subtract(x, y)
    return np.dot(a.T, a)


def Endmember_extract(x, p):  # SiVM 方法
    [D, N] = x.shape
    Z1 = np.zeros((1, 1))
    O1 = np.ones((1, 1))
    d = np.zeros((p, N))
    I = np.zeros((p, 1))
    V = np.zeros((1, N))
    ZD = np.zeros((D, 1))
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), ZD)

    I = np.argmax(d[0, :])

    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I].reshape(D, 1))

    for v in range(1, p):
        D1 = np.concatenate((d[0:v, I].reshape((v, I.size)), np.ones((v, 1))), axis=1)
        D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
        D4 = np.concatenate((D1, D2), axis=0)
        D4 = np.linalg.inv(D4)
        for i in range(N):
            D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
            V[0, i] = np.dot(np.dot(D3.T, D4), D3)

        I = np.append(I, np.argmax(V))
        for i in range(N):
            d[v, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I[v]].reshape(D, 1))

    per = np.argsort(I)
    I = np.sort(I)
    d = d[per, :]
    return I, d


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


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


def plotEndmembersAndGT(endmembers, endmembersGT, endmember_path, sadsave, save):
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
        save.append(endmembers[SAD_index[i], :])

    sadsave.append(SAD_endmember.mean())
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.86)
    plt.savefig(endmember_path + '.png')
    plt.draw()
    plt.pause(0.1)
    plt.close()


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
    plt.draw()
    plt.pause(0.1)
    plt.close()


def plotAbundancesGT(abundanceGT, abundance_path):
    num_endmembers = abundanceGT.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    title = '参照丰度图'
    cmap = 'viridis'
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





