import sys
import scipy as sp
import scipy.linalg as splin
import tensorflow as tf
from tensorflow.python import keras
from keras.layers import Input, Conv2D, BatchNormalization, SpatialDropout2D, GaussianNoise
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import Model, Sequential, layers, optimizers, activations, callbacks
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import io as sio
from skimage.transform.pyramids import pyramid_reduce
from skimage.transform import rescale
import keras.backend as K
import matplotlib.pyplot as plt
import warnings
import matplotlib
from PIL import Image
from sklearn.metrics import mean_squared_error
import csv

matplotlib.rc("font", family='Microsoft YaHei')
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import pandas as pd


class HSI:
    '''
    A class for Hyperspectral Image (HSI) data.
    '''

    def __init__(self, data, rows, cols, gt, abundance_gt):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()
        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows

        # padding
        patch_size = 32
        image = np.reshape(data, (self.rows, self.cols, self.bands))
        h = rows
        w = cols
        h1 = h // patch_size if h // patch_size == 0 else h // patch_size + 1
        w1 = w // patch_size if w // patch_size == 0 else w // patch_size + 1
        image_pad = np.pad(image, ((0, patch_size * h1 - h), (0, patch_size * w1 - w), (0, 0)), 'edge')

        self.image = np.reshape(data, (self.rows, self.cols, self.bands))
        self.image_pad = image_pad
        self.gt = gt
        self.abundance_gt = abundance_gt

    def array(self):
        """返回 像元*波段 的数据阵列（array)

        Returns:
            a matrix -- array of spectra
        """
        return np.reshape(self.image, (self.rows * self.cols, self.bands))

    def get_bands(self, bands):
        return self.image[:, :, bands]

    def crop_image(self, start_x, start_y, delta_x=None, delta_y=None):
        if delta_x is None: delta_x = self.cols - start_x
        if delta_y is None: delta_y = self.rows - start_y
        self.cols = delta_x
        self.rows = delta_y
        self.image = self.image[start_x:delta_x + start_x, start_y:delta_y + start_y, :]
        return self.image

    """crop_image:这是一个图片裁剪的函数。它接受四个参数，其中start_x和start_y表示裁剪的起始位置，delta_x和delta_y表示裁剪的宽度和高度。如果delta_x和delta_y没有指定，
       则默认为从起始位置到图片末尾的距离。在函数内部，首先检查delta_x和delta_y是否有值，如果没有，则计算出它们应该具有的默认值。
       然后将self.cols和self.rows属性设置为新的值，因为裁剪后的图片大小会发生改变。最后，使用NumPy数组切片语法对图片进行裁剪，并返回裁剪后的图片。
       这个函数可以用于图像处理任务中，例如将一张大图裁剪成多个小图进行分析和处理。"""


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


class SumToOne(layers.Layer):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.num_outputs = params['num_endmembers']
        self.params = params

    def l_regularization(self, x):
        patch_size = self.params['patch_size'] * self.params['patch_size']
        z = tf.abs(x + tf.keras.backend.epsilon())
        # l_half = tf.reduce_sum(tf.norm(z, self.params['l'], axis=3), axis=None)
        l_half = tf.reduce_sum(tf.norm(z, 1, axis=3), axis=None)
        return 1.0 / patch_size * self.params['l1'] * l_half

    def tv_regularization(self, x):
        patch_size = self.params['patch_size'] * self.params['patch_size']
        # z = tf.abs(x + tf.keras.backend.epsilon())
        tv = tf.reduce_sum(tf.image.total_variation(x))
        return 1.0 / patch_size * self.params['tv'] * tv

    def call(self, x):
        if self.params['l1'] > 0.0:
            self.add_loss(self.l_regularization(x))
        if self.params['tv'] > 0.0:
            self.add_loss(self.tv_regularization(x))
        return x


class Scaling(layers.Layer):
    def __init__(self, params, **kwargs):
        super(Scaling, self).__init__(**kwargs)
        self.params = params

    def non_zero(self, x):
        patch_size = self.params['patch_size'] * self.params['patch_size']
        # z = tf.abs(x + tf.keras.backend.epsilon())
        tv = tf.reduce_sum(tf.image.total_variation(x))
        return 1.0 / patch_size * self.params['tv'] * tv

    def call(self, x):
        self.add_loss(self.tv_regularization(x))
        return K.relu(x)


def SAD(y_true, y_pred):
    y_true2 = tf.math.l2_normalize(y_true, axis=-1)
    y_pred2 = tf.math.l2_normalize(y_pred, axis=-1)
    A = tf.keras.backend.mean(y_true2 * y_pred2)
    sad = tf.math.acos(A)
    return sad


# 正确的应该是A=(y_true*y_pred)/(y_true2 * y_pred2)  sad=tf.math.acos(A)  这里的SAD是当做网络中的损失函数，这么写可能是为了简化
# 下面的numpy_SAD才是计算SAD评价指标的
# 还有一种写法是A=(y_true2 * y_pred2)

def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos > 1.0: cos = 1.0
    return np.arccos(cos)


# 端元对应和计算对应端元的SAD
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


def numpy_MSE(y_true, y_pred):  # 错写成MSE了，实际上是算RMSE，将错就错了
    num_cols = y_pred.shape[0]
    num_rows = y_pred.shape[1]
    diff = y_true - y_pred
    squared_diff = np.square(diff)
    mse = squared_diff.sum() / (num_rows * num_cols)
    rmse = np.sqrt(mse)
    return rmse


def calRE(y_pr, y_tr):
    armse_y = np.sqrt(np.mean(np.mean((y_pr - y_tr) ** 2, axis=1)))
    return armse_y


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


def plotEndmembers(endmembers):
    if len(endmembers.shape) > 2 and endmembers.shape[1] > 1:
        endmembers = np.squeeze(endmembers).mean(axis=0).mean(axis=0)
    else:
        endmembers = np.squeeze(endmembers)
    num_endmembers = np.min(endmembers.shape)
    fig = plt.figure(num=1, figsize=(8, 8))
    n = num_endmembers / 2
    if num_endmembers % 2 != 0: n = n + 1
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[i, :], 'r', linewidth=1.0)
        ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('endm.png')
    plt.close()


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

        # 这行代码是用于隐藏Matplotlib图形中X轴的刻度和标签，使得图形中只显示Y轴的刻度和标签。
        # 具体地说，ax是一个AxesSubplot对象，get_xaxis()方法返回X轴的Axis对象。set_visible(False)方法为该Axis对象设置可见性为False
    sadsave.append(SAD_endmember.mean())
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.86)
    plt.savefig(endmember_path + '.png')


def plotAbundancesSimple(abundances, abundanceGT, abundance_path, rmsesave):
    abundances = np.transpose(abundances, axes=[1, 0, 2])  # 把行列颠倒，第三维不动，因为方法代码里写的得到的丰度是列*行，但是如果行列数相同，倒也不影响
    num_endmembers = abundances.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    abundance_index, MSE_abundance = order_abundance(abundanceGT, abundances)
    print(abundance_index)
    title = "RMSE: " + np.array2string(MSE_abundance.mean(),
                                       formatter={'float_kind': lambda x: "%.3f" % x})
    cmap = 'jet'
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

        """代码片段设置了colormap（颜色映射）“cmap”为“viridis”，创建了一个大小为12x12的图形，对“abundances”数组沿着最后一个轴进行求和，
        然后对“abundances”数组的第一个轴进行迭代以创建子图。对于每个子图，它创建一个新的axes对象，
        并使用“make_axes_locatable（）”函数创建一个新的“cax”对象用于颜色条。然后，它在“ax”对象上调用“imshow（）”，
        并使用来自“abundances”数组的第“i”个切片和指定的colormap“cmap”。
        最后，它使用“plt.colorbar（）”添加颜色条，并使用之前创建的“im”对象和“cax”对象。
        它还使用“ax.get_xaxis（）。set_visible（False）”和“ax.get_yaxis（）。set_visible（False）”隐藏每个子图的x和y轴标签。
        总的来说，这段代码可能用于可视化某种多维数据，其中“abundances”数组中的每个切片表示数据的不同方面。颜色条有助于提供数组中值的视觉比例。
        特定选择的colormap“viridis”是一种感知均匀的颜色映射，常用于科学可视化。"""
    rmsesave.append(MSE_abundance.mean())
    plt.tight_layout()  # 用于自动调整子图参数，以便使所有子图适合整个图像区域，并尽可能地减少子图之间的重叠
    plt.rcParams.update({'font.size': 15})
    plt.suptitle(title)
    plt.subplots_adjust(top=0.91)
    plt.savefig(abundance_path + '.png')


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


class PlotWhileTraining(callbacks.Callback):
    def __init__(self, plot_every_n, hsi):
        super(PlotWhileTraining, self).__init__()
        self.plot_every_n = plot_every_n
        self.input = hsi.array()
        self.cols = hsi.cols
        self.rows = hsi.rows
        self.endmembersGT = hsi.gt
        self.num_endmembers = hsi.gt.shape[0]
        self.sads = None
        self.epochs = []
        self.RE = None

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.sads = []
        self.RE = []

    def get_re(self):
        endmembers = self.model.layers[-1].get_weights()[0]
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("abundances").output
        )
        abundances1 = intermediate_layer_model.predict(self.input)
        abundances = np.reshape(abundances1,
                                [self.cols, self.rows, self.num_endmembers])
        recon = self.reconstruct(abundances, endmembers)
        self.RE = np.sqrt(np.mean(np.mean((recon - self.input) ** 2, axis=1)))
        return self.epochs, self.RE

    def on_batch_end(self, batch, logs={}):
        return

    def reconstruct(self, S, A):
        S = np.reshape(S, (S.shape[0] * S.shape[1], S.shape[2]))
        reconstructed = np.matmul(S, A)
        return reconstructed

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('SAD'))
        self.num_epochs = epoch
        self.epoch, self.RE, = self.get_re()

        if self.plot_every_n == 0 or epoch % self.plot_every_n != 0:
            return
        else:
            print(self.RE)


def reconstruct(S, A):
    S = np.reshape(S, (S.shape[0] * S.shape[1], S.shape[2]))
    reconstructed = np.matmul(S, A)
    return reconstructed


def estimate_snr(Y, r_m, x):
    # L number of bands (channels), N number of pixels
    [L, N] = Y.shape
    [p, N] = x.shape  # p number of endmembers (reduced dimension)

    P_y = sp.sum(Y ** 2) / float(N)
    P_x = sp.sum(x ** 2) / float(N) + sp.sum(r_m ** 2)
    snr_est = 10 * sp.log10((P_x - p / L * P_y) / (P_y - P_x))

    return snr_est


def vca(Y, R, verbose=True, snr_input=0):
    # Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    #
    # ------- Input variables -------------
    #  Y - matrix with dimensions L(channels) x N(pixels)
    #      each pixel is a linear mixture of R endmembers
    #      signatures Y = M x s, where s = gamma x alfa
    #      gamma is a illumination perturbation factor and
    #      alfa are the abundance fractions of each endmember.
    #  R - positive integer number of endmembers in the scene
    #
    # ------- Output variables -----------
    # Ae     - estimated mixing matrix (endmembers signatures)
    # indice - pixels that were chosen to be the most pure
    # Yp     - Data matrix Y projected.
    #
    # ------- Optional parameters---------
    # snr_input - (float) signal to noise ratio (dB)
    # v         - [True | False]
    # ------------------------------------

    # Initializations

    if len(Y.shape) != 2:
        sys.exit(
            'Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

    [L, N] = Y.shape  # L number of bands (channels), N number of pixels

    R = int(R)
    if (R < 0 or R > L):
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')

    # SNR Estimates

    if snr_input == 0:
        y_m = sp.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m  # data with zero-mean
        # computes the R-projection matrix
        Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :R]
        # project the zero-mean data onto p-subspace
        x_p = sp.dot(Ud.T, Y_o)

        SNR = estimate_snr(Y, y_m, x_p)

        if verbose:
            print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10 * sp.log10(R)

    # Choosing Projective Projection or
    #          projection to p-1 subspace

    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")

            d = R - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = sp.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                # computes the p-projection matrix
                Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :d]
                # project thezeros mean data onto p-subspace
                x_p = sp.dot(Ud.T, Y_o)

            Yp = sp.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  # x_p =  Ud.T * Y_o is on a R-dim subspace
            c = sp.amax(sp.sum(x ** 2, axis=0)) ** 0.5
            y = sp.vstack((x, c * sp.ones((1, N))))
    else:
        if verbose:
            print("... Select the projective proj.")

        d = R
        # computes the p-projection matrix
        Ud = splin.svd(sp.dot(Y, Y.T) / float(N))[0][:, :d]

        x_p = sp.dot(Ud.T, Y)
        # again in dimension L (note that x_p has no null mean)
        Yp = sp.dot(Ud, x_p[:d, :])

        x = sp.dot(Ud.T, Y)
        u = sp.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
        y = x / sp.dot(u.T, x)

    # VCA algorithm

    indice = sp.zeros(R, dtype=int)
    A = sp.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = sp.rand(R, 1)
        f = w - sp.dot(A, sp.dot(splin.pinv(A), w))
        f = f / splin.norm(f)

        v = sp.dot(f.T, y)

        indice[i] = sp.argmax(sp.absolute(v))
        A[:, i] = y[:, indice[i]]  # same as x(:,indice(i))

    Ae = Yp[:, indice]

    return Ae, indice, Yp
