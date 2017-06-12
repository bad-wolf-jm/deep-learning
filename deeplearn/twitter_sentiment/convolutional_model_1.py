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
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.core import Flatten, Activation
from keras.layers import Dense, Input, Dropout
from keras.layers.normalization import BatchNormalization

def convolutional_block_part(input_, output_channels):
    x = Conv1D(output_channels, kernel_size = [3], strides=1, padding='causal', use_bias = True)(input_)
    x = BatchNormalization(axis=2)(x)
    return Activation('relu')(x)

def convolutional_block(input_, output_channels):
    x = convolutional_block_part(input_, output_channels)
    x = convolutional_block_part(x, output_channels)
    return x

input_  = Input(shape = [144], dtype='uint16')

layer_1 = Embedding(256, 16, input_length = 144)(input_)
layer_1 = Conv1D(64, kernel_size = [3], strides=1, padding='causal', use_bias = True)(layer_1)

layer_2 = convolutional_block(layer_1, 64)
layer_2 = convolutional_block(layer_2, 64)
layer_2 = MaxPooling1D(pool_size= 2, strides=2)(layer_2)

layer_3 = convolutional_block(layer_2, 128)
layer_3 = convolutional_block(layer_3, 128)
layer_3 = MaxPooling1D(pool_size= 2, strides=2)(layer_3)

layer_4 = convolutional_block(layer_3, 256)
layer_4 = convolutional_block(layer_4, 256)
layer_4 = MaxPooling1D(pool_size= 2, strides=2)(layer_4)

layer_5 = convolutional_block(layer_4, 512)
layer_5 = convolutional_block(layer_5, 512)
layer_5 = MaxPooling1D(pool_size= 3, strides=2)(layer_5)

decision_layer = Flatten()(layer_5)
decision_layer = Dense(2048, activation='linear')(decision_layer)
decision_layer = Dense(512, activation='linear')(decision_layer)
decision_layer = Dense(2, activation='softmax')(decision_layer)

model = Model(inputs = input_, outputs=decision_layer)

print(model.summary())
