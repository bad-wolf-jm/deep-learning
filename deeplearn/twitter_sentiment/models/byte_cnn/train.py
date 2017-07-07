#from convolutional_model_1 import model
from models.byte_cnn.byte_cnn import ByteCNN
from models.tf_session import tf_session
import tensorflow as tf
#import os
#import glob
import numpy as np
#import html
#import time
import sys
#import zipfile
import signal
#import sys
#import pymysql
from stream.receiver import DataReceiver

max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 140

def pad(array, length):
    array = list(array[:length])
    array += [0] * (length - len(array))
    return array


def save_before_exiting(*a):
    print(a)
    print("Process is being killed, I'm saving the weights")
    model.save_weights(MODEL_WEIGHT_KILLED_FILE)
    print("Model weights saved to", MODEL_WEIGHT_KILLED_FILE)
    print("Exiting")
    sys.exit(0)

batch_index = 0
train_batch_losses = []
train_batch_accuracies = []
validation_batch_losses = []
validation_batch_accuracies = []


def train_on_batch(train_x, train_y):
    global batch_index, train_batch_losses, train_batch_accuracies
    batch_x = np.array([pad(element, MAX_TWEET_LENGTH) for element in train_x])
    batch_y = np.array([element for element in train_y])
    d = model.train(batch_x, batch_y)
    batch_index += 1
    train_batch_losses.append([batch_index, float(d['loss'])])
    train_batch_accuracies.append([batch_index, float(d['accuracy'])])
    print('train', d)

def validate_on_batch(train_x, train_y):
    global batch_index, validation_batch_losses, validation_batch_accuracies
    batch_x = np.array([pad(element, MAX_TWEET_LENGTH) for element in train_x])
    batch_y = np.array(train_y)
    d = model.validate(batch_x, batch_y)
    validation_batch_losses.append([batch_index, float(d['loss'])])
    validation_batch_accuracies.append([batch_index, float(d['accuracy'])])
    print('validate', d)

def get_train_batch_data(min_batch_index = None, max_batch_index = None):
    if len(train_batch_losses) > 0:
        min_batch_index = min_batch_index or 0
        max_batch_index = max_batch_index or max([x[0] for x in train_batch_losses])
        batch_losses = [x for x in train_batch_losses if x[0]>= min_batch_index and x[0] <= max_batch_index]
        batch_accuracies = [x for x in train_batch_accuracies if x[0]>= min_batch_index and x[0] <= max_batch_index]
        return {'losses': batch_losses, 'accuracies':batch_accuracies}
    else:
        return {'losses': [], 'accuracies':[]}

def get_validation_batch_data(min_batch_index = None, max_batch_index = None):
    if len(validation_batch_losses) > 0:
        min_batch_index = min_batch_index or 0
        max_batch_index = max_batch_index or max([x[0] for x in validation_batch_losses])
        batch_losses = [x for x in validation_batch_losses if x[0]>= min_batch_index and x[0] <= max_batch_index]
        batch_accuracies = [x for x in validation_batch_accuracies if x[0]>= min_batch_index and x[0] <= max_batch_index]
        return {'losses': batch_losses, 'accuracies':batch_accuracies}
    else:
        return {'losses': [], 'accuracies':[]}


model = ByteCNN()
model.build_training_model()
model.initialize()


with tf_session() as session:
    foo = DataReceiver()
    bar = DataReceiver(port=99887)
    session.run(tf.global_variables_initializer())
    foo.register_action_handler('train', train_on_batch)
    foo.register_action_handler('validate', validate_on_batch)
    bar.register_action_handler('get_train_batch_data', get_train_batch_data)
    bar.register_action_handler('get_validation_batch_data', get_validation_batch_data)
    bar.start(True)
    foo.start(False)

print("Training done!!!")
print("Writing the model's weights to 'data/convolutional_character_model.hd5'")
# model.save_weights('data/convolutional_character_model.hd5')
