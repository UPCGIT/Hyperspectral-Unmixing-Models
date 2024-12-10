import tensorflow as tf
from tensorflow.keras import initializers, constraints, layers, activations, regularizers
from tensorflow.python.keras import backend as K
from unmixing import vca, reconstruct
from unmixing import plotEndmembersAndGT, plotAbundancesSimple, load_HSI, plotAbundancesGT
from scipy import io as sio
import os
import numpy as np
from numpy.linalg import inv
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import time

warnings.filterwarnings("ignore")

starttime = time.time()


# OSP方法
def OSP(B, R):
    dots = 0.0
    B = tf.linalg.l2_normalize(B, axis=0)
    for i in range(R):
        for j in range(i + 1, R):
            A1 = B[:, i]
            A2 = B[:, j]
            dot = tf.reduce_sum(A1 * A2, axis=0)
            dots = dots + dot
    return dots


class SumToOne(layers.Layer):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.num_outputs = params['num_endmembers']
        self.params = params

    def l1_regularization(self, x):
        l1 = tf.reduce_sum(tf.pow(tf.abs(x) + 1e-8, 0.7))
        return self.params['l1'] * l1

    def osp_regularization(self, x):
        return self.params['osp'] * OSP(x, self.params['num_endmembers'])

    def call(self, x):
        x = tf.nn.softmax(self.params['scale'] * x)
        self.add_loss(self.l1_regularization(x))
        self.add_loss(self.osp_regularization(x))
        return x


class NonNegLessOne(regularizers.Regularizer):
    def __init__(self, strength):
        super(NonNegLessOne, self).__init__()
        self.strength = strength

    def __call__(self, x):
        neg = tf.cast(x < 0, x.dtype) * x
        greater_one = tf.cast(x >= 1.0, x.dtype) * x
        reg = -self.strength * tf.reduce_sum(neg) + self.strength * tf.reduce_sum(greater_one)
        return reg


class HyperLaplacianLoss(object):
    def __init__(self, scale):
        super(HyperLaplacianLoss).__init__()
        self.scale = scale

    def loss(self, X, R):
        fidelity = tf.reduce_mean(tf.pow(tf.abs(X - R) + tf.keras.backend.epsilon(), 0.7), axis=None)
        x = tf.linalg.l2_normalize(X, axis=1)
        r = tf.linalg.l2_normalize(R, axis=1)
        s = X.get_shape().as_list()
        log_cosines = tf.reduce_sum(tf.math.log(tf.reduce_sum(r * x, axis=1) + K.epsilon()))
        return self.scale * fidelity - log_cosines


class Autoencoder(object):
    def __init__(self, params, W=None):
        self.data = params["data"].array()
        self.params = params
        self.decoder = layers.Dense(
            units=self.params["n_bands"],
            kernel_regularizer=NonNegLessOne(10),
            activation='linear',
            name="output",
            use_bias=False,
            kernel_constraint=None)
        self.hidden1 = layers.Dense(
            units=self.params["num_endmembers"],
            activation=self.params["activation"],
            name='hidden1',
            use_bias=True
        )
        self.hidden2 = layers.Dense(
            units=self.params["num_endmembers"],
            activation='linear',
            name='hidden2',
            use_bias=True
        )

        self.asc_layer = SumToOne(self.params, name='abundances')
        self.model = self.create_model()
        self.initalize_encoder_and_decoder(W)
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def initalize_encoder_and_decoder(self, W):
        if W is None: return
        self.model.get_layer('output').set_weights([W.T])
        self.model.get_layer('hidden1').set_weights([W, np.zeros(self.params["num_endmembers"])])
        W2 = inv(np.matmul(W.T, W))
        self.model.get_layer('hidden2').set_weights([W2, np.zeros(self.params["num_endmembers"])])

    def create_model(self):
        input_features = layers.Input(shape=(self.params["n_bands"],))
        code = self.hidden1(input_features)
        code = self.hidden2(code)
        code = layers.BatchNormalization()(code)
        abunds = self.asc_layer(code)
        output = self.decoder(abunds)

        return tf.keras.Model(inputs=input_features, outputs=output)

    def fix_decoder(self):
        for l in self.model.layers:
            l.trainable = True
        self.model.layers[-1].trainable = False
        self.decoder.trainable = False
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def fix_encoder(self):
        for l in self.model.layers:
            l.trainable = True
        self.model.get_layer('hidden1').trainable = False
        self.model.get_layer('hidden2').trainable = False
        self.hidden1.trainable = False
        self.hidden2.trainable = False
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def fit(self, data):
        return self.model.fit(
            x=data,
            y=data,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"])

    def train_alternating(self, data, epochs):
        for epoch in range(epochs):
            self.fix_decoder()
            self.model.fit(x=data, y=data,
                           batch_size=self.params["batch_size"],
                           epochs=2)
            self.fix_encoder()
            self.model.fit(x=data, y=data,
                           batch_size=self.params["batch_size"],
                           epochs=1)

    # 交替训练的过程中，首先固定解码器，只训练编码器，然后固定编码器，只训练解码器   fix_ 是用来固定的
    def get_endmembers(self):
        return self.model.layers[-1].get_weights()[0]

    def get_abundances(self):
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("abundances").output
        )
        abundances1 = intermediate_layer_model.predict(self.data)
        abundances = np.reshape(abundances1,
                                [self.params['data'].cols, self.params['data'].rows, self.params['num_endmembers']])

        return abundances, abundances1


class OutlierDetection(object):
    def __init__(self, image, alpha, threshold):
        self.I = image
        self.alpha = alpha
        self.threshold = threshold

    def get_neighbors(self, row, column):
        n, m, b = self.I.shape
        neighbors_x = np.s_[max(row - 1, 0):min(row + 1, n - 1) + 1]
        neighbors_y = np.s_[max(column - 1, 0):min(column + 1, m - 1) + 1]
        block = np.zeros((3, 3, b))
        block_x = np.s_[max(row - 1, 0) - row + 1:min(row + 1, n - 1) + 1 - row + 1]
        block_y = np.s_[max(column - 1, 0) - column + 1:min(column + 1, m - 1) + 1 - column + 1]
        block[block_x, block_y] = self.I[neighbors_x, neighbors_y, :]
        block = np.reshape(block, (9, -1))
        block = np.delete(block, 5, 0)
        return block

    def d(self, x, y):
        return np.linalg.norm(x - y) ** 2

    def s(self, row, column):
        N = self.get_neighbors(row, column)
        x0 = self.I[row, column, :]
        dists = list(map(lambda x: self.d(x0, x), N))
        return 1 / 8 * sum(list(map(lambda x: np.exp(-x / self.alpha), dists)))

    def create_heatmap(self):
        n, m, b = self.I.shape
        M = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                M[j, i] = self.s(i, j)
        return M

    def get_training_data(self):
        M = self.create_heatmap()
        maxM = np.max(M.flatten())
        indices = np.argwhere(M > self.threshold)
        # M[M<self.threshold] = 0
        arr = np.zeros((indices.shape[0], self.I.shape[2]))
        i = 0
        for [r, c] in indices:
            arr[i, :] = self.I[r, c, :]
            i = i + 1
        return [arr, M]


datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'dc': 'DC2',
                'sim': 'sim30'}
dataset = "sim"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")

IsOutlierDetection = False

# Hyperparameters
num_endmembers = hsi.gt.shape[0]
num_spectra = 2000
batch_size = 15
learning_rate = 0.001
epochs = 13
n_bands = hsi.bands

opt = tf.optimizers.Adam(learning_rate=learning_rate)
activation = 'relu'
l1 = 1.0
osp = 0.5

# hsi.gt=None

if IsOutlierDetection:
    data, hmap = OutlierDetection(hsi.image, 0.05, 0.25).get_training_data()
    plt.figure(figsize=(12, 12))
    plt.imshow(hmap, cmap='gray')
    plt.colorbar()
    plt.show()
    num_spectra = data.shape[0]
    batch_size = 256
else:
    data = hsi.array()

fid_scale = batch_size
loss = HyperLaplacianLoss(fid_scale).loss

# Hyperparameter dictionary
params = {
    "activation": activation,
    "num_endmembers": num_endmembers,
    "batch_size": batch_size,
    "num_spectra": num_spectra,
    "data": hsi,
    "epochs": epochs,
    "n_bands": n_bands,
    "GT": hsi.gt,
    "lr": learning_rate,
    "optimizer": opt,
    "loss": loss,
    "scale": 1,
    "l1": l1,
    "osp": osp
}

num_runs = 5
results_folder = 'D:/毕设/Results'
method_name = 'OSPAEU'

# Dictonary of aliases for datasets. The first string is the key and second is value (name of matfile without .mat
# suffix) Useful when looping over datasets


for dataset in [dataset]:
    save_folder = results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
    abundance_folder = results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
    endmember_folder = results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(abundance_folder):
        os.makedirs(abundance_folder)
    if not os.path.exists(endmember_folder):
        os.makedirs(endmember_folder)

    hsi = load_HSI(
        "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
    true = hsi.array()
    hsi.image = hsi.image - np.min(hsi.image, axis=2, keepdims=True) + 0.000001  # negative values cause trouble
    data, hmap = OutlierDetection(hsi.image, 0.005, 0.1).get_training_data()

    num_spectra = data.shape[0]
    batch_size = 256
    params['num_spectra'] = num_spectra
    params['data'] = hsi
    params['n_bands'] = hsi.bands
    a = []
    b = []
    r = []

    for run in range(1, num_runs + 1):
        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        params['optimizer'] = opt
        training_data = data[np.random.randint(0, data.shape[0], num_spectra), :]
        save_name = datasetnames[dataset] + '_run' + str(run) + '.mat'
        save_path = save_folder + '/' + save_name
        abundance_name = datasetnames[dataset] + '_run' + str(run)
        abundance_path = abundance_folder + '/' + abundance_name
        abundanceGT_path = results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
            dataset] + '参照丰度图'
        endmember_name = datasetnames[dataset] + '_run' + str(run)
        endmember_path = endmember_folder + '/' + endmember_name
        vca_end = vca(data.T, num_endmembers)[0]
        autoencoder = Autoencoder(params, vca_end)
        autoencoder.train_alternating(training_data, epochs)
        endmembers = autoencoder.get_endmembers()
        abundances, abundance2dim = autoencoder.get_abundances()
        plotEndmembersAndGT(endmembers, hsi.gt, endmember_path, a)
        plotAbundancesSimple(abundances, hsi.abundance_gt, abundance_path, b)
        sio.savemat(save_path, {'M': endmembers, 'A': abundances})
        recon = reconstruct(abundances, endmembers)
        RE = np.sqrt(np.mean(np.mean((recon - true) ** 2, axis=1)))
        r.append(RE)

a = np.reshape(a, (-1, num_endmembers + 1))  # 最后一列是mSAD
b = np.reshape(b, (-1, num_endmembers + 1))
dt = pd.DataFrame(a)
dt2 = pd.DataFrame(b)
dt3 = pd.DataFrame(r)
dt.to_csv(results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各端元SAD及mSAD运行结果.csv')
dt2.to_csv(results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '各丰度图RMSE及mRMSE运行结果.csv')
dt3.to_csv(results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
    dataset] + '重构误差RE运行结果.csv')
plotAbundancesGT(hsi.abundance_gt, abundanceGT_path)
endtime = time.time()
total_time = endtime - starttime
print("程序运行时间为：", total_time, "秒")
