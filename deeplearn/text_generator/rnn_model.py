import time, os, sys

import keras
import numpy as np
#import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot

from models.batch_generator import simple_generator
from models.progress_display import basic_callback

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import Dense, Input, concatenate, Lambda
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras import losses
from keras import metrics
from keras import backend as K

BATCH_SIZE    = 100
SEQ_LEN       = 30
ALPHABET_SIZE = 128
INTERNAL_SIZE = 512


input_         = Input(shape = [SEQ_LEN], batch_shape = [BATCH_SIZE, SEQ_LEN], dtype='uint8') #[BATCH_SIZE, SEQ_LEN]
input_one_hot  = Lambda(K.one_hot, arguments={'num_classes': ALPHABET_SIZE}, output_shape=[SEQ_LEN, ALPHABET_SIZE])(input_)


lay = GRU(INTERNAL_SIZE, batch_input_shape = (BATCH_SIZE, SEQ_LEN, ALPHABET_SIZE), activation = 'relu', use_bias=True, dropout=0.2, recurrent_dropout=0.2, stateful = True, return_sequences = True)(input_one_hot)
lay = GRU(INTERNAL_SIZE, batch_input_shape = (BATCH_SIZE, SEQ_LEN, ALPHABET_SIZE), activation = 'relu', use_bias=True, dropout=0.2, recurrent_dropout=0.2, stateful = True, return_sequences = True)(lay)
lay = GRU(INTERNAL_SIZE, batch_input_shape = (BATCH_SIZE, SEQ_LEN, ALPHABET_SIZE), activation = 'relu', use_bias=True, dropout=0.2, recurrent_dropout=0.2, stateful = True, return_sequences = True)(lay)
predictions = TimeDistributed(Dense(ALPHABET_SIZE, activation='softmax'))(lay)

output_ = Input(shape = [SEQ_LEN], batch_shape = [BATCH_SIZE, SEQ_LEN], dtype='uint8') #[BATCH_SIZE, SEQ_LEN]
output_one_hot  = Lambda(K.one_hot, arguments={'num_classes': ALPHABET_SIZE}, output_shape=[SEQ_LEN, ALPHABET_SIZE])(output_)


model_output = keras.layers.concatenate([output_one_hot, predictions], axis = 1)
#loss = losses.categorical_crossentropy(output_one_hot, predictions)
model = Model(input = [input_, output_], outputs = model_output)

def loss_function(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_pred[:, :SEQ_LEN, :], y_pred[:, SEQ_LEN:, :])
    return loss

def accuracy(y_true, y_pred):
    loss = metrics.categorical_accuracy(y_pred[:, :SEQ_LEN, :], y_pred[:, SEQ_LEN:, :])
    return loss
