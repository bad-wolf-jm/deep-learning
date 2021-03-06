import time, os, sys

import keras
import numpy as np
#import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot

#from models.batch_generator import simple_generator
#from models.progress_display import basic_callback

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.core import Flatten, Activation
from keras.layers import Dense, Input, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def convolutional_block_part(input_, output_channels):
    x = Conv1D(output_channels, kernel_size = [3], strides=1, padding='causal', use_bias = True)(input_)
    x = BatchNormalization(axis=1)(x)
    return Activation('relu')(x)

def convolutional_block(input_, output_channels):
    x = convolutional_block_part(input_, output_channels)
    x = convolutional_block_part(x, output_channels)
    return x


CNN_LEVEL_1_FEATURES = 64
CNN_LEVEL_2_FEATURES = 128
CNN_LEVEL_3_FEATURES = 256
CNN_LEVEL_4_FEATURES = 512

INPUT_LENGTH = 140

input_  = Input(shape = [INPUT_LENGTH], dtype='uint8')
#layer_1  = Lambda(K.one_hot, arguments={'num_classes': 128}, output_shape=[INPUT_LENGTH, 128])(input_)
layer_1 = Embedding(128, 16, input_length = INPUT_LENGTH)(input_)
layer_1 = Conv1D(64, kernel_size = [3], strides=1, padding='causal', use_bias = True)(layer_1)

layer_2 = convolutional_block(layer_1, CNN_LEVEL_1_FEATURES)
layer_2 = convolutional_block(layer_2, CNN_LEVEL_1_FEATURES)
layer_2 = convolutional_block(layer_2, CNN_LEVEL_1_FEATURES)
layer_2 = MaxPooling1D(pool_size= 2, strides=2)(layer_2)

layer_3 = convolutional_block(layer_2, CNN_LEVEL_2_FEATURES)
layer_3 = convolutional_block(layer_3, CNN_LEVEL_2_FEATURES)
layer_3 = convolutional_block(layer_3, CNN_LEVEL_2_FEATURES)
layer_3 = MaxPooling1D(pool_size= 2, strides=2)(layer_3)

layer_4 = convolutional_block(layer_3, CNN_LEVEL_3_FEATURES)
layer_4 = convolutional_block(layer_4, CNN_LEVEL_3_FEATURES)
layer_4 = convolutional_block(layer_4, CNN_LEVEL_3_FEATURES)
layer_4 = MaxPooling1D(pool_size= 2, strides=2)(layer_4)

layer_5 = convolutional_block(layer_4, CNN_LEVEL_4_FEATURES)
layer_5 = convolutional_block(layer_5, CNN_LEVEL_4_FEATURES)
layer_5 = convolutional_block(layer_5, CNN_LEVEL_4_FEATURES)
layer_5 = MaxPooling1D(pool_size= 3, strides=2)(layer_5)

decision_layer = Flatten()(layer_5)
decision_layer = Dense(2048, activation='linear', use_bias = True)(decision_layer)
decision_layer = Dropout(0.4)(decision_layer)
decision_layer = Dense(2048, activation='linear', use_bias = True)(decision_layer)
decision_layer = Dropout(0.4)(decision_layer)
decision_layer = Dense(2, activation='softmax')(decision_layer)
model = Model(inputs = input_, outputs=decision_layer)

print(model.summary())
