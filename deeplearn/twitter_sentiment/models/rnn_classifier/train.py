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
from train.supervisor import TrainingSupervisor


from stream.nn.streamer import TrainingDataStreamer
from models.rnn_classifier.sentiment import generate_batches, flags, count_rows
#
#
#import zmq
import argparse
#import pymysql
import numpy as np
from config import db, stream


#host, port = flags.stream_to.split(':')
#port = int(port)
#streamer = TrainingDataStreamer(validation_interval=flags.validation_interval, summary_span=None)

N = count_rows()
test = N // 100
batch_generator = generate_batches(min_id=test + 1, batch_size=flags.batch_size, epochs=flags.epochs)
validation_iterator = generate_batches(min_id=0, max_id=test, batch_size=flags.validation_size, epochs=None)
#streamer.stream(batch_generator, validation_iterator, host=host, port=port)
#try:
#    streamer.stream(batch_generator, validation_iterator, host=host, port=port)
#except KeyboardInterrupt:
#    streamer.streamer.shutdown()



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


    def test_model(self, train_x, train_y):
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
foo = TrainRNNClassifier(model, flags.validation_interval)
#foo.run_training()
def save_before_exiting(*a):
    path = foo.save_model_image()
    foo.shutdown()
    print('\rProcess terminated, model saved as', path)

signal.signal(signal.SIGTERM, save_before_exiting)

try:
    foo.run_training(batch_generator, validation_iterator)
except KeyboardInterrupt:
    save_before_exiting()
    foo.shutdown()
    sys.exit(0)
