from keras import backend as K
import tensorflow as tf


def SAD(y_true, y_pred):
    y_true2 = tf.math.l2_normalize(y_true, axis=-1)
    y_pred2 = tf.math.l2_normalize(y_pred, axis=-1)
    A = tf.keras.backend.mean(y_true2 * y_pred2)
    sad = tf.math.acos(A)
    return sad


def normSAD(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    sad = SAD(y_true, y_pred)
    return 0.008 * mse + 1.0 * sad


def normMSE(y_true, y_pred):
    y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    mse = K.mean(K.square(y_true - y_pred))
    return mse


def normSAD2(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    sad = SAD(y_true, y_pred)
    return 0.005 * mse + 0.75 * sad


