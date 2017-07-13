#from convolutional_model_1 import model
from models.tweet2vec.lstm_cnn_autoencode import Tweet2Vec_LSTM
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
from stream.nn.trainer import TrainingSupervisor


max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 512

class TrainAutoencoder(TrainingSupervisor):
    def train_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_y])
        d = self.model.train(batch_x, batch_y)
        print(d)
        return d

    def validation_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_y])
        d = self.model.validate(batch_x, batch_y)
        return d

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array

model = Tweet2Vec_LSTM()
model.build_training_model()
model.initialize()
foo = TrainAutoencoder(model)

def save_before_exiting(*a):
    path = foo.save_model_image()
    foo.shutdown()
    print('\rProcess terminated, model saved as', path)

signal.signal(signal.SIGTERM, save_before_exiting)

try:
    foo.run_training()
except KeyboardInterrupt:
    save_before_exiting()
    foo.shutdown()
    sys.exit(0)
