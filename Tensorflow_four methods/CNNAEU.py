import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from scipy import io as sio
import numpy as np
from unmixing import plotEndmembersAndGT, plotAbundancesSimple, SAD, load_HSI, plotAbundancesGT, reconstruct
import os
import time
from sklearn.feature_extraction.image import extract_patches_2d
import pandas as pd
disable_eager_execution()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

starttime = time.time()


def training_input_fn(hsi, patch_size, patch_number, batch_size):
    patches = extract_patches_2d(hsi, (patch_size, patch_size), max_patches=patch_number)
    return patches


class SumToOne(tf.keras.layers.Layer):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.num_outputs = params['num_endmembers']
        self.params = params

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

    def call(self, x):
        x = tf.nn.softmax(self.params['scale'] * x)
        return x


class Encoder(tf.keras.Model):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.hidden_layer_one = tf.keras.layers.Conv2D(filters=self.params['e_filters'],
                                                       kernel_size=self.params['e_size'],
                                                       activation=self.params['activation'], strides=1, padding='same',
                                                       kernel_initializer=params['initializer'], use_bias=False)
        self.hidden_layer_two = tf.keras.layers.Conv2D(filters=self.params['num_endmembers'], kernel_size=1,
                                                       activation=self.params['activation'], strides=1, padding='same',
                                                       kernel_initializer=self.params['initializer'], use_bias=False)
        self.asc_layer = SumToOne(params=self.params, name='abundances')

    def call(self, input_patch):
        code = self.hidden_layer_one(input_patch)
        code = tf.keras.layers.BatchNormalization()(code)
        code = tf.keras.layers.SpatialDropout2D(0.2)(code)
        code = self.hidden_layer_two(code)
        code = tf.keras.layers.BatchNormalization()(code)
        code = tf.keras.layers.SpatialDropout2D(0.2)(code)
        code = self.asc_layer(code)
        return code


class Decoder(tf.keras.layers.Layer):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.output_layer = tf.keras.layers.Conv2D(filters=params['d_filters'], kernel_size=params['d_size'],
                                                   activation='linear',
                                                   kernel_constraint=tf.keras.constraints.non_neg(),
                                                   name='endmembers', strides=1, padding='same',
                                                   kernel_regularizer=None,
                                                   kernel_initializer=params['initializer'], use_bias=False)

    def call(self, code):
        recon = self.output_layer(code)
        return recon

    def getEndmembers(self):
        return self.output_layer.get_weights()


class Autoencoder(tf.keras.Model):
    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.params = params

    def call(self, patch):
        abunds = self.encoder(patch)
        # tf.summary.histogram('abunds', abunds, step=1)
        #         abunds = tf.keras.layers.SpatialDropout2D(0.08)(abunds)
        output = self.decoder(abunds)
        return output

    def getEndmembers(self):
        endmembers = self.decoder.getEndmembers()[0]
        if endmembers.shape[1] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=0).mean(axis=0)
        else:
            endmembers = np.squeeze(endmembers)
        return endmembers

    def getAbundances(self, hsi):
        return np.squeeze(self.encoder.predict(np.expand_dims(hsi, 0)))  # 得到的丰度自然的是三维的行*列*波段

    def train(self, patches):
        self.fit(patches, patches, epochs=self.params['epochs'], batch_size=self.params['batch_size'], verbose=1)
    # verbose :是否显示这样的详细信息。1为显示，默认显示。0为不显示。 Epoch 4/4
    # 250/250 [==============================] - 0s 2ms/sample - loss: 1.5649


def loss(model, original):
    reconstruction_error = SAD(model(original), original)
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original) + sum(model.losses), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)


# Hyperparmameter settings
# 数据集别名字典。第一个字符串是键，第二个字符串是值.在数据集上循环时有用
datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'dc': 'DC2',
                'sim': 'sim30'}
dataset = "Samson"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
n_bands = hsi.gt.shape[1]
num_endmembers = hsi.gt.shape[0]
hsi = hsi.image  # load_HSI得到的是波段*像元的，而该方法要用到的是行*列*波段的高光谱数据，所以要利用hsi.image才能到行*列*波段
patch_size = 40
num_patches = 250
batch_size = 15
learning_rate = 0.003
epochs = 320

scale = 3  # scaling for softmax
l2 = 0
l1 = 0e-8
tv = 0e-8

activation = tf.keras.layers.LeakyReLU(0.02)
initializer = tf.keras.initializers.RandomNormal(0.0, 0.3)
regularizer = tf.keras.regularizers.l2(l2)

opt = tf.optimizers.RMSprop(learning_rate=learning_rate, decay=0.0)

params = {'e_filters': 48,  # e_filters is the number of featuremaps in the first hidden layer
          'e_size': 3,  # e_size is the size of the hidden layer's filter
          'd_filters': n_bands,
          'd_size': 13,  # d_size is the decoder's filter size
          'activation': activation,
          'num_endmembers': num_endmembers,
          'scale': scale,
          'regularizer': regularizer,
          'initializer': initializer,
          'l1': l1,
          'tv': tv,
          'patch_size': patch_size,
          'batch_size': batch_size,
          'num_patches': num_patches,
          'data': hsi,
          'epochs': epochs}

num_runs = 1
results_folder = 'D:/毕设/Results'
method_name = 'CNNAEU'

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
    data = hsi.image
    true = hsi.array()
    a = []
    b = []
    r = []
    for i in range(num_runs):
        print('Run number: ' + str(i + 1))
        save_name = datasetnames[dataset] + '_run' + str(i) + '.mat'
        save_path = save_folder + '/' + save_name
        abundance_name = datasetnames[dataset] + '_run' + str(i)
        abundance_path = abundance_folder + '/' + abundance_name
        abundanceGT_path = results_folder + '/' + method_name + '/' + datasetnames[dataset] + '/' + datasetnames[
            dataset] + '参照丰度图'
        endmember_name = datasetnames[dataset] + '_run' + str(i)
        endmember_path = endmember_folder + '/' + endmember_name
        patches = training_input_fn(data, patch_size, num_patches, batch_size)
        autoencoder = Autoencoder(params)
        autoencoder.compile(opt, loss=SAD)
        autoencoder.train(patches=patches)
        endmembers = autoencoder.getEndmembers()
        abundances = autoencoder.getAbundances(data)
        plotAbundancesSimple(abundances, hsi.abundance_gt, abundance_path, b)
        plotEndmembersAndGT(endmembers, hsi.gt, endmember_path, a)
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
