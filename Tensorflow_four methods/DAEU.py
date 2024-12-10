import tensorflow as tf
from tensorflow.keras import initializers, constraints, layers, activations, regularizers
from tensorflow.python.keras import backend as K
from unmixing import HSI, plotEndmembers, SAD, order_endmembers
from unmixing import plotEndmembersAndGT, plotAbundancesSimple, load_HSI, plotAbundancesGT
from unmixing import calRE, order_abundance, PlotWhileTraining, reconstruct, vca
from scipy import io as sio
import os
import numpy as np
import warnings
from tensorflow.keras.callbacks import TensorBoard
import time
import pandas as pd

warnings.filterwarnings("ignore")

starttime = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 和为一约束   self是把类作为方法调用，后面会看到def fit  然后后面用的是autoencoder.fit
class SumToOne(layers.Layer):
    def __init__(self, **kwargs):
        super(SumToOne, self).__init__(**kwargs)

    def call(self, x):
        x *= K.cast(x >= K.epsilon(), K.floatx())  # 布尔值0or1用cast转为float型
        x = K.relu(x)
        x = x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon())
        return x


# 软阈值ReLU激活函数
class SparseReLU(tf.keras.layers.Layer):
    def __init__(self, params):
        self.params = params
        super(SparseReLU, self).__init__()
        self.alpha = self.add_weight(shape=(self.params['num_endmembers'],), initializer=tf.keras.initializers.Zeros(),
                                     trainable=True, constraint=tf.keras.constraints.non_neg())

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=input_shape[1:], initializer=tf.keras.initializers.Zeros(),
                                     trainable=True, constraint=tf.keras.constraints.non_neg())
        super(SparseReLU, self).build(input_shape)

    def call(self, x):
        return tf.keras.backend.relu(x - self.alpha)



# 自编码器结构
class Autoencoder(object):
    def __init__(self, params):
        self.data = None
        self.params = params
        self.is_deep = True
        self.model = self.create_model()
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def create_model(self):
        use_bias = False
        n_end = self.params['num_endmembers']
        # Input layer
        Sparse_ReLU = SparseReLU(self.params)
        input_ = layers.Input(shape=(self.params['n_bands'],))

        encoded = layers.Dense(n_end * 9, use_bias=use_bias, activation=self.params['activation'])(input_)
        # 括号的意思是将括号里的上一层东西传给这一层
        encoded = layers.Dense(n_end * 6, use_bias=use_bias, activation=self.params['activation'])(encoded)
        # encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dense(n_end * 3, use_bias=use_bias, activation=self.params['activation'])(encoded)
        # encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dense(n_end, use_bias=use_bias, activation=self.params['activation'])(encoded)

        # Batch Normalization
        encoded = layers.BatchNormalization()(encoded)
        # Soft Thresholding
        encoded = Sparse_ReLU(encoded)
        # Sum To One (ASC)
        encoded = SumToOne(name='abundances')(encoded)

        # Gaussian Dropout
        decoded = layers.GaussianDropout(0.0045)(encoded)

        # Decoder
        decoded = layers.Dense(self.params['n_bands'], activation='linear', name='endmembers',
                               use_bias=False,
                               kernel_constraint=constraints.non_neg())(encoded)  # 这里括号里是encoded，表示不使用上面的高斯dropot层

        return tf.keras.Model(inputs=input_, outputs=decoded)

    def fit(self, data, plot_every):
        plot_callback = PlotWhileTraining(plot_every, self.params['data'])
        return self.model.fit(
            x=data,
            y=data,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"], callbacks=[plot_callback])  # callbacks=[tensorboard_callback]

        # 这段代码定义了一个名为fit的方法，其中包含三个参数：data、plot_every和self。data参数是传递给模型的训练数据，x=data,y=data是因为这段代码中使用的是自监督学习
        # plot_every参数表示训练过程中每隔多少个epoch就会绘制一次图表。self参数指向当前实例对象。
        # 在方法的内部，首先创建了一个名为plot_callback的对象，该对象是一个PlotWhileTraining类的实例，用于在训练过程中绘制图表。接着，使用self.model.fit()
        # 方法来拟合数据。该方法中传递了一些参数，包括训练数据、批大小、迭代次数和回调函数列表（这里只包含了plot_callback对象）。最后，返回模型拟合的结果。
        # 总的来说，这段代码定义了一个训练模型的方法，其中使用了回调函数来实时监控训练过程中的性能，并在训练过程结束后返回模型的拟合结果。

    def get_endmembers(self):
        return self.model.layers[-1].get_weights()[0]

    def get_abundances(self):
        intermediate_layer_model = tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("abundances").output
        )
        abundances1 = intermediate_layer_model.predict(self.params['data'].array())
        abundances = np.reshape(abundances1,
                                [self.params['data'].cols, self.params['data'].rows, self.params['num_endmembers']])
        # 最终，该方法返回一个三维的Numpy数组，其中第一维和第二维分别对应着输入数据的列数和行数，第三维则对应着端元的数量
        return abundances, abundances1


# 设置超参数和加载高光谱数据
datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'sim': 'sim1020',
                'dc': 'DC2',
                'berlin': 'berlin_test',
                'apex': 'apex_new',
                'moni': 'moni30',
                'houston': 'houston_test',
                'moffett': 'moffett'}
dataset = "houston"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")

# num_endmembers = 4  # 记得改端元数量
num_endmembers = hsi.gt.shape[0]
num_spectra = 2500  # 从data数组中随机选择num_spectra个样本，随机采样数据集;moffett:500
batch_size = 6
learning_rate = 0.002
epochs = 40
loss = SAD  # 损失函数是SAD
opt = tf.optimizers.RMSprop(learning_rate=learning_rate)
data = hsi.array()

# Hyperparameter dictionary
params = {
    "num_endmembers": num_endmembers,
    "batch_size": batch_size,
    "num_spectra": num_spectra,
    "data": hsi,
    "epochs": epochs,
    "n_bands": hsi.bands,
    "GT": hsi.gt,
    "lr": learning_rate,
    "optimizer": opt,
    "loss": loss,
    "activation": layers.LeakyReLU(0.1),
    "cols": hsi.cols,
    "rows": hsi.rows
}

plot_every = 0
num_runs = 10
results_folder = 'D:/毕设/Results'
method_name = 'DAEU'
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
    data = hsi.array()
    true = hsi.image
    a = []
    b = []
    r = []
    for run in range(1, num_runs + 1):
        training_data = data[np.random.randint(0, data.shape[0], num_spectra), :]
        """这行代码使用NumPy库中的random.randint()函数从data数组中随机选择num_spectra个样本，并将其存储在training_data数组中。data.shape[0]
        表示data数组的行数，也就是样本的数量。使用随机采样的方式获取训练数据可以帮助避免过拟合，提高模型的泛化能力。当然也可以不随机采样，将全部的数据用于训练"""

        params['opt'] = tf.optimizers.RMSprop(learning_rate=learning_rate)
        save_name = datasetnames[dataset] + '_run' + str(run) + '.mat'
        save_path = save_folder + '/' + save_name
        abundance_name = datasetnames[dataset] + '_run' + str(run)
        abundance_path = abundance_folder + '/' + abundance_name
        abundanceGT_path = results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
            dataset] + '参照丰度图'
        endmember_name = datasetnames[dataset] + '_run' + str(run)
        endmember_path = endmember_folder + '/' + endmember_name
        autoencoder = Autoencoder(params)

        # 获取模型的所有可训练变量
        trainable_vars = autoencoder.model.count_params()
        # tensorboard_callback = TensorBoard(log_dir='./logs', update_freq='epoch')
        autoencoder.fit(training_data, plot_every)
        endmembers = autoencoder.get_endmembers()
        abundances, abundance2dim = autoencoder.get_abundances()
        plotEndmembersAndGT(endmembers, hsi.gt, endmember_path, a)
        plotAbundancesSimple(abundances, hsi.abundance_gt, abundance_path, b)
        sio.savemat(save_path, {'M': endmembers, 'A': abundances})
        recon = reconstruct(abundances, endmembers)
        RE = np.sqrt(np.mean(np.mean((recon - data) ** 2, axis=1)))
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
