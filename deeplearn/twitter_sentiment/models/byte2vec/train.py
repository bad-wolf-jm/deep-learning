#from convolutional_model_1 import model
from models.rnn_classifier.bidirectional_gru import Tweet2Vec_BiGRU
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
MAX_TWEET_LENGTH = 1024

class TrainRNNClassifier(TrainingSupervisor):
    def train_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([element for element in train_y])
        d = self.model.train(batch_x, batch_y)
        print(d)
        return d

    def validation_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([element for element in train_y])
        d = self.model.validate(batch_x, batch_y)
        return d

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array

model = Tweet2Vec_BiGRU()
model.build_training_model()
model.initialize()
foo = TrainRNNClassifier(model)
foo.run_training()
