from __future__ import print_function
import time
from keras.constraints import non_neg
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from losses import SAD, normSAD
from datetime import datetime
from scipy.io import loadmat, savemat
import tensorflow as tf
from utility import SparseReLU, SumToOne
from utility import load_HSI, plotEndmembersAndGT, plotAbundancesSimple, plotAbundancesGT
import os
import random
import pandas as pd
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

start_time = time.time()
# batchsize若过小，耗时长,训练效率低。
# 假设batchsize=1,每次用一个数据进行训练,如果数据总量很多时(假设有十万条数据),就需要向模型投十万次数据,完整训练完一遍数据需要很长的时问,训练效率很低;
# 原来这里写的batchsize是20，一个epoch要6min
latent_vec = None

datasetnames = {'Urban': 'Urban4',
                'Jasper': 'Jasper',
                'Samson': 'Samson',
                'synthetic': 'synthetic5',
                'dc': 'DC2',
                'apex': 'apex_new',
                'moni': 'moni30',
                'houston': 'houston_test',
                'moffett': 'moffett'}
dataset = "houston"
hsi = load_HSI(
    "D:/毕设/Datasets/" + datasetnames[dataset] + ".mat")
x = hsi.array()
E = hsi.gt
end = []
abu = []
r = []

s_gt = hsi.abundance_gt
num_cols = hsi.abundance_gt.shape[0]
s_gt = np.reshape(s_gt, [num_cols*num_cols, -1])

num_runs = 10
latent_dim = num_endmembers = hsi.gt.shape[0]
epochs = 10
batchsize = x.shape[0] // 10
l_vca = 0.01
use_bias = False
activation_set = LeakyReLU(0.2)
initializer = tf.initializers.glorot_normal()


def E_reg(weight_matrix):
    return l_vca * SAD(weight_matrix, E)


def create_model(input_dim, latent_dim):
    autoencoder_input = Input(shape=(input_dim,))
    generator_input = Input(shape=(input_dim,))
    encoder = Sequential()
    encoder.add(Dense(latent_dim * 9, input_shape=(input_dim,), activation=activation_set, name='Dense_1'))
    encoder.add(Dense(latent_dim * 6, activation=activation_set, name='Dense_2'))
    encoder.add(Dense(latent_dim * 3, activation=activation_set, name='Dense_3'))
    encoder.add(Dense(latent_dim, use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                      activation=activation_set))
    encoder.add(BatchNormalization())
    # Soft Thresholding
    encoder.add(SparseReLU(alpha_initializer='zero', alpha_constraint=non_neg(), activity_regularizer=None))
    # Sum To One (ASC)
    encoder.add(SumToOne(axis=0, name='abundances', activity_regularizer=None))

    decoder = Sequential()
    decoder.add(Dense(input_dim, input_shape=(latent_dim,), activation='linear', name='endmembers', use_bias=use_bias,
                      kernel_constraint=non_neg(), kernel_regularizer=E_reg, kernel_initializer=initializer))

    discriminator = Sequential()
    discriminator.add(Dense(intermediate_dim2, input_shape=(latent_dim,), activation='relu'))
    discriminator.add(Dense(intermediate_dim1, activation='relu'))
    discriminator.add(Dense(1, activation='sigmoid'))

    autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
    autoencoder.compile(optimizer=Adam(learning_rate=2e-4), loss=normSAD)

    discriminator.compile(optimizer=Adam(learning_rate=2e-4), loss="binary_crossentropy")
    discriminator.trainable = False
    generator = Model(generator_input, discriminator(encoder(generator_input)))
    generator.compile(optimizer=Adam(learning_rate=2e-4), loss="binary_crossentropy")

    """
    print("Autoencoder Architecture")
    print(autoencoder.summary())
    print("Discriminator Architecture")
    print(discriminator.summary())
    print("Generator Architecture")
    print(generator.summary())
    # 可输出模型结构与参数
    """

    """
    plot_model(autoencoder, to_file="autoencoder_graph.png")
    plot_model(discriminator, to_file="discriminator_graph.png")
    plot_model(generator, to_file="generator_graph.png")
    可绘制模型结构图
    """

    return autoencoder, discriminator, generator, encoder, decoder


def train(batch_size, n_epochs):
    autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=hsi.gt.shape[1], latent_dim=latent_dim)
    for epoch in np.arange(1, n_epochs + 1):
        autoencoder_losses = []
        discriminator_losses = []
        generator_losses = []

        for batch in range(x.shape[0] // batch_size):
            start = int(batch * batch_size)
            end = int(start + batch_size)
            samples = x[start:end]

            autoencoder_history = autoencoder.fit(x=samples, y=samples, epochs=30, batch_size=batch_size,
                                                  validation_split=0.0, verbose=0)

            fake_latent = encoder.predict(samples)
            real_sample = s_gt[start:end]
            real_sample = np.random.normal(size=(batch_size, latent_dim))
            discriminator_input = np.concatenate((fake_latent, real_sample))
            discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=10,
                                                      batch_size=batch_size, validation_split=0.0, verbose=0)
            generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=10,
                                              batch_size=batch_size, validation_split=0.0, verbose=0)
            autoencoder_losses.append(autoencoder_history.history["loss"])

            discriminator_losses.append(discriminator_history.history["loss"])
            generator_losses.append(generator_history.history["loss"])
        if (epoch + 1) % 2 == 0:
            print("\nEpoch {}/{} ".format(epoch+1, n_epochs, ))
            print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))

            print("Discriminator Loss: {}".format(np.mean(discriminator_losses)))
            print("Generator Loss: {}".format(np.mean(generator_losses)))

    z_latent = encoder.predict(x)
    endmember = decoder.get_weights()[0]
    reconstruction = decoder.predict(z_latent)
    return z_latent, endmember, reconstruction


if __name__ == "__main__":
    global desc, intermediate_dim1, intermediate_dim2, intermediate_dim3
    original_dim = hsi.gt.shape[1]
    intermediate_dim1 = int(np.ceil(original_dim * 1.2) + 5)
    intermediate_dim2 = int(max(np.ceil(original_dim / 4), latent_dim + 2) + 3)
    intermediate_dim3 = int(max(np.ceil(original_dim / 10), latent_dim + 1))

    desc = "aae"



    output_path = 'D:/毕设/Results'
    method_name = 'AAeNet'
    mat_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'mat'
    endmember_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'endmember'
    abundance_folder = output_path + '/' + method_name + '/' + datasetnames[dataset] + '/' + 'abundance'
    if not os.path.exists(mat_folder):
        os.makedirs(mat_folder)
    if not os.path.exists(endmember_folder):
        os.makedirs(endmember_folder)
    if not os.path.exists(abundance_folder):
        os.makedirs(abundance_folder)

    for run in range(num_runs):
        """
        random_seed = 1
        random.seed(random_seed)  # set random seed for python
        np.random.seed(random_seed)  # set random seed for numpy
        tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
        os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu
        """
        endmember_name = datasetnames[dataset] + '_run' + str(run)
        endmember_path = endmember_folder + '/' + endmember_name
        abundance_name = datasetnames[dataset] + '_run' + str(run)
        abundance_path = abundance_folder + '/' + abundance_name

        print('Start Running! run:', run+1)

        z_latent, endmember, re = train(batch_size=batchsize, n_epochs=epochs)
        z_latent = np.reshape(z_latent, [num_cols, num_cols, -1])
        plotEndmembersAndGT(endmember, hsi.gt, endmember_path, end)
        plotAbundancesSimple(z_latent, hsi.abundance_gt, abundance_path, abu)
        armse_y = np.sqrt(np.mean(np.mean((re - x) ** 2, axis=1)))
        r.append(armse_y)
        sio.savemat(mat_folder + '/' + method_name + '_run' + str(run) + '.mat', {'EM': endmember,
                                                                                  'A': z_latent,
                                                                                  'Y_hat': re
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
    print('程序运行时间为：', end_time - start_time, 's')

