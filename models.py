import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, BatchNormalization, LeakyReLU, LSTM, Attention
from keras_self_attention import SeqSelfAttention
import keras.backend as K
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


def build_LSTM_multihead(nb_features, sequence_length, nb_out):
    in_layers, out_layers = list(), list()
    for i in range(nb_features):
      inputs = Input(shape=(sequence_length,1))
      rnn1 = Conv1D(filters=64,kernel_size=2,strides=1,padding="same")(inputs)
      lr1= LeakyReLU()(rnn1)
      bn1= BatchNormalization()(lr1)
      rnn2 = Conv1D(filters=64,kernel_size=2,strides=1,padding="same")(bn1)
      lr2= LeakyReLU()(rnn2)
      bn2= BatchNormalization()(lr2)
      rnn3 = LSTM(units=50, return_sequences=True)(bn2)
      lr3= LeakyReLU()(rnn3)
      bn3= BatchNormalization()(lr3)
      att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            bias_regularizer=keras.regularizers.l1(1e-4),
                            attention_regularizer_weight=1e-4,
                            attention_width=15)(bn3)
      pool1 = MaxPooling1D(pool_size=2)(att)
      flat = Flatten()(pool1)
      # store layers
      in_layers.append(inputs)
      out_layers.append(flat)
    # merge heads
    merged = concatenate(out_layers)
    # interpretation
    dense1 = Dense(50, activation='relu')(merged)
    outputs = Dense(nb_out, activation='relu')(dense1)
    model = Model(inputs=in_layers, outputs=outputs)

    return model


def build_LSTM_simple(time_steps, columns, nb_out):

    model = keras.Sequential([
        layers.LSTM(128, input_shape = (time_steps, len(columns)), return_sequences=True, activation = "tanh"),
        layers.LSTM(64, activation = "tanh", return_sequences = True),
        layers.LSTM(32, activation = "tanh"),
        layers.Dense(96, activation = "relu"),
        layers.Dense(128, activation = "relu"),
        layers.Dense(nb_out)
    ])


    return model


def build_CNN(columns, nb_out):
    keras.backend.clear_session()
    model = keras.Sequential()
    model.add(Conv1D(32,kernel_size = 2,activation='relu',input_shape=(28,len(columns))))
    model.add(Conv1D(128,kernel_size = 2,activation='relu'))
    model.add(Conv1D(128,kernel_size = 2,activation='relu'))
    model.add(Conv1D(64,kernel_size = 2,activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(nb_out))

    return model


def build_NN(input_shape, stddev_guess=1):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    x = layers.LSTM(128, input_shape = input_shape, return_sequences=True, activation = "tanh")(x)
    x = layers.LSTM(64, activation = "tanh", return_sequences = True)(x)
    x = layers.LSTM(32, activation = "tanh")(x)
    x = layers.Dense(96, activation = "relu")(x)
    x = layers.Dense(128, activation = "relu")(x)
    mu_logsigma = layers.Dense(2, activation='linear')(x) # output layer, which produces the mean and the log of the sigma (out_shape = 2)
    lf = lambda t: tfp.distributions.Normal(loc=t[:, :1], scale=tf.math.exp(t[:, 1:]))
    # loc=t[:, :1] --> extract the mean
    # exp(t[:, 1:]) --> extract the std dev
    model_out = tfp.layers.DistributionLambda(lf)(mu_logsigma)
    model = keras.Model(model_in, model_out)

    return model


def k_score(y_true, y_pred):

    """score metric used for model evaluation"""

    d = y_pred - y_true
    return K.sum(K.exp(d[d >= 0] / 10) - 1) + K.sum(K.exp(-1 * d[d < 0] / 13) - 1)


def rmse(y_true, y_pred):

    """rmse metric used for model evaluation"""

    return K.sqrt(K.mean(K.square(y_pred - y_true)))
