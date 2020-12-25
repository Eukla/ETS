import time
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from ets.algorithms.MLSTM.utils.generic_utils import load_dataset_at
from ets.algorithms.MLSTM.utils.keras_utils import train_model, evaluate_model, set_trainable, loss_model
from ets.algorithms.MLSTM.utils.layer_utils import AttentionLSTM
import os
DATASET_INDEX = 0

TRAINABLE = True


def generate_model(size,cell=8):
    f = open("./ets/algorithms/MLSTM/utils/constants.txt", "r")
    lines = f.read()
    lines = lines.split("\n")
    MAX_NB_VARIABLES = 0
    NB_CLASSES_LIST = 0
    MAX_TIMESTEPS_LIST = 0
    for line in lines:
        if "VARIABLES" in line:
            line = line.split("=")
            MAX_NB_VARIABLES = int(line[1])
        if "CLASSES" in line:
            line = line.split("=")
            NB_CLASSES_LIST = int(line[1])
        if "TIMESTEPS" in line:
            line = line.split("=")
            MAX_TIMESTEPS_LIST = int(line[1])
    f.close()
    MAX_NB_VARIABLES = MAX_NB_VARIABLES
    NB_CLASS = NB_CLASSES_LIST
    ip = Input(shape=(MAX_NB_VARIABLES, size))

    x = Masking()(ip)
    x = AttentionLSTM(cell)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


def run():
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX,
                                                                      fold_index=None,
                                                                      normalize_timeseries=False)
    start = time.time()
    model = generate_model(len(X_test[0][0]),8)
    train_model(model, DATASET_INDEX, dataset_prefix="current", epochs=600, batch_size=128)
    train = time.time() - start
    res = evaluate_model(model, DATASET_INDEX, dataset_prefix="current", batch_size=128)
    test = time.time() - start
    return res, train, test