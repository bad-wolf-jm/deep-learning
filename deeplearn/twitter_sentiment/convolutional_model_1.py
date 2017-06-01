import time, os, sys

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot

from models.batch_generator import simple_generator
from models.progress_display import basic_callback

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten, Dense


model = Sequential()
model.add(Embedding(256, 5, input_length = 144))
model.add(Conv1D(10, kernel_size = 5, strides=1, padding='causal', activation='relu', use_bias=True))
model.add(MaxPooling1D(pool_size=1, strides=None, padding='valid'))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(2, activation = 'softmax'))
print(model.summary())


if __name__ == '__main__':

    model_folder = 'model_structure'
    model_name   = 'convolutional_model.json'

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    with open(os.path.join(model_folder, model_name), 'w') as model_file:
        model_file.write(model.to_json())
