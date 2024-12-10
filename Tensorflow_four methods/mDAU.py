import tensorflow as tf
from tensorflow.keras import initializers, constraints, layers, activations, regularizers
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape
from unmixing import HSI, plotEndmembers, PlotWhileTraining, vca, SAD
from unmixing import plotEndmembersAndGT, plotAbundancesSimple, load_HSI, plotAbundancesGT, reconstruct
from scipy import io as sio
import os
import numpy as np
import warnings
import pandas as pd
import time

warnings.filterwarnings("ignore")

starttime = time.time()



class SumToOne(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.params = params

    def call(self, x):
        x = K.abs(x) / (K.sum(K.abs(x), axis=-1, keepdims=True) + K.epsilon())
        return x


class mDA_initializer(tf.keras.initializers.Initializer):
    def __init__(self, data, p):
        self.W = self.mDA(data.T, p)

    def mDA(self, X, p):
        X = np.vstack((X, np.ones((1, X.shape[1]))))
        d = X.shape[0]
        q = np.vstack((np.ones((d - 1, 1)) * (1 - p), np.ones(1)))
        S = np.matmul(X, X.T)
        Q = S * np.matmul(q, q.T)
        row, col = np.diag_indices_from(Q)
        Q[row, col] = np.multiply(q.T, np.diag(S))
        P = np.multiply(S, np.repeat(q.T, d, 0))
        a = P[0:-1, :]
        b = Q + 1e-5 * np.eye(d)
        W = np.linalg.lstsq(b.T, a.T, rcond=None)[0].T
        return W.astype(np.float32)

    def __call__(self, shape, dtype=None):
        return tf.constant(value=self.W)

    # call函数，用于在调用类的实例时返回初始化的权重矩阵，利用tf.constant将self.W转换为TensorFlow的常量，并返回给调用者。

    def get_config(self):  # To support serialization
        return {"W": self.W}
    # 这里定义了一个get_config函数，用于支持序列化。它将self.W打包成一个字典并返回，以便在需要将类的实例转换为JSON或其他格式时使用。


class NonNeg(regularizers.Regularizer):
    def __init__(self, strength):
        super(NonNeg, self).__init__()
        self.strength = strength

    def __call__(self, x):
        neg = tf.abs(tf.cast(x < 0, x.dtype) * x)  # 布尔值 大于零的变成零不管，小于零的变成true,cast转为1，再乘以自己取绝对值
        reg = self.strength * tf.reduce_sum(neg)  # strength 自定义的正则化强度
        return reg


class DenseTied(tf.keras.layers.Layer):
    def __init__(
            self,
            units,
            activation=None,
            use_bias=False,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            tied_to=None,
            **kwargs
    ):
        self.tied_to = tied_to
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self.non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

        # 如果该层与另一个层共享权重，则将其它层的权重矩阵转置后作为该层的权重矩阵,且加到不可训练项中；
        # 否则，根据输入数据的维度(input_dim)和神经元数量(units)构建一个新的权重矩阵。同时，也可以选择是否使用偏置，并且对权重矩阵和偏置向量进行正则化和约束。

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    # 根据输入数据的形状(input_shape)和神经元数量(units)，计算该层的输出数据的形状(output_shape)。
    # 其中，输出数据的维度与输入数据的维度相同，只有最后一个维度的大小改变为神经元数量。

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format="channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output
    # 定义了该层的前向传播过程，即将输入数据(inputs)与权重矩阵(kernel)相乘，再加上偏置向量(bias)（如果使用偏置），并将结果输入激活函数进行激活，最终得到输出数据(output)。


class AugmentedLogistic(tf.keras.layers.Layer):
    def __init__(self):
        super(AugmentedLogistic, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # 这个断言语句的意思是，如果输入数据的维度数小于2，即数据的rank小于2，就会抛出一个AssertionError异常，提示输入数据的形状不符合要求
        input_dim = input_shape[-1]
        self.a = self.add_weight(shape=(input_dim,),
                                 initializer="ones",
                                 trainable=True)
        self.b = self.add_weight(shape=(input_dim,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, x):
        y = self.a * x + self.b
        return tf.nn.sigmoid(y)


class Autoencoder(object):
    def __init__(self, params):
        self.data = params["data"].array()
        self.params = params
        self.mDA_layer = layers.Dense(
            units=self.params["n_bands"] + 1,
            activation="linear",
            use_bias=False,
            kernel_initializer=mDA_initializer(self.data, self.params["p"]),
        )
        self.unmix_layer = layers.Dense(
            units=self.params["num_endmembers"],
            activation="linear",
            kernel_regularizer=NonNeg(10),
            name='unmix',
            use_bias=False
        )
        self.output_layer = DenseTied(
            units=self.params["n_bands"],
            kernel_constraint=None,
            activation="linear",
            tied_to=self.unmix_layer,
        )
        self.asc_layer = SumToOne(self.params, name='abundances')
        self.model = self.create_model()
        init = vca(data.T, params['num_endmembers'])[0]
        self.model.get_layer(name='unmix').set_weights([init])  # vca初始化
        self.model.compile(optimizer=self.params["optimizer"], loss=self.params["loss"])

    def create_model(self):
        input_features = layers.Input(shape=(self.params["n_bands"],))
        code = self.mDA_layer(input_features)
        code = layers.Lambda(lambda x: x[:, :-1])(code)
        # 这行代码使用了Keras中的Lambda层，它将输入的张量进行裁剪，选择从第一维开始到倒数第二维的所有元素，相当于去掉了输入张量的最后一维
        code = self.unmix_layer(code)
        code = AugmentedLogistic()(code)
        abunds = self.asc_layer(code)
        output = self.output_layer(abunds)
        return tf.keras.Model(inputs=input_features, outputs=output)

    def fit(self, data):
        return self.model.fit(
            x=data,
            y=data,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"], )

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


datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson'}
dataset = "Samson"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
data = hsi.array()
# Hyperparameters
num_endmembers = hsi.gt.shape[0]
num_spectra = 2000
batch_size = 6
learning_rate = 0.001
epochs = 10
loss = SAD
opt = tf.optimizers.RMSprop(learning_rate=learning_rate, momentum=0.9)

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
    "p": 0.01,
    "optimizer": opt,
    "loss": loss,
}

num_runs = 10
results_folder = 'D:/毕设/Results'
method_name = 'mDAU'

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
        "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat"
    )
    data = hsi.array()
    true = hsi.image
    a = []
    b = []
    r = []
    for run in range(1, num_runs + 1):
        training_data = data[np.random.randint(0, data.shape[0], num_spectra), :]
        save_name = datasetnames[dataset] + '_run' + str(run) + '.mat'
        save_path = save_folder + '/' + save_name
        abundance_name = datasetnames[dataset] + '_run' + str(run)
        abundance_path = abundance_folder + '/' + abundance_name
        abundanceGT_path = results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
            dataset] + '参照丰度图'
        endmember_name = datasetnames[dataset] + '_run' + str(run)
        endmember_path = endmember_folder + '/' + endmember_name
        autoencoder = Autoencoder(params)
        autoencoder.fit(training_data)
        endmembers = autoencoder.get_endmembers()
        abundances, abundances2dim = autoencoder.get_abundances()
        plotEndmembersAndGT(endmembers.T, hsi.gt, endmember_path, a)
        plotAbundancesSimple(abundances, hsi.abundance_gt, abundance_path, b)
        sio.savemat(save_path, {'M': endmembers, 'A': abundances})
        recon = reconstruct(abundances, endmembers.T)
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
