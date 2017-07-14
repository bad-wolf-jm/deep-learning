#from convolutional_model_1 import model
from models.byte2vec.byte2vec import Byte2Vec
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


max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 1024

# def train_on_batch(train_x, train_y):
#    query = [x[0] for x in train_x]
#    context = [x[1] for x in train_x]
#    training_values = trainer.train(np.array(query), np.array(context), np.array(train_y))
#    foo = "Loss = {loss:.4f} --- Accuracy = {accuracy:.4f}".format(**training_values)
#    return training_values
#
# def validate_on_batch(train_x, train_y):
#    training_values = trainer.validate(np.array(train_x))
#    foo = training_values
#    return training_values
#
#
#

from stream.nn.streamer import TrainingDataStreamer
from models.byte2vec.skipgram import generate_batches, flags
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

#N = count_rows()
#test = N // 100
batch_generator = generate_batches(batch_size=flags.batch_size, epochs=flags.epochs, noise_ratio=10, window_size=5, num_skips=2)
#validation_iterator = generate_batches(min_id=0, max_id=test, batch_size=flags.validation_size, epochs=None)
#streamer.stream(batch_generator, host=host, port=port)
# try:
#    streamer.stream(batch_generator, host=host, port=port)
# except KeyboardInterrupt:
#    streamer.streamer.shutdown()


class TrainByte2Vec(TrainingSupervisor):
    def train_step(self, train_x, train_y):
        query = [x[0] for x in train_x]
        context = [x[1] for x in train_x]
        training_values = self.model.train(np.array(query), np.array(context), np.array(train_y))
        foo = "Loss = {loss:.4f} --- Accuracy = {accuracy:.4f}".format(**training_values)
        print (foo)
        return training_values
#
#
#        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
#        batch_y = np.array([element for element in train_y])
#        d = self.model.train(batch_x, batch_y)
#        print(d)
#        return d

    def validation_step(self, train_x, train_y):
        training_values = self.model.validate(np.array(train_x))
        foo = training_values
        # print(foo)
        return training_values

    def test_model(self, train_x, train_y):
        training_values = self.model.validate(np.array(train_x))
        foo = training_values
        # print(foo)
        return training_values

#        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
#        #batch_y = np.array([element for element in train_y])
#        d = self.model.validate(batch_x, batch_y)
#        return d

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array


model = Byte2Vec()
model.build_training_model()
model.initialize()
foo = TrainByte2Vec(model, flags.validation_interval)
# foo.run_training()


def save_before_exiting(*a):
    path = foo.save_model_image()
    foo.shutdown()
    print('\rProcess terminated, model saved as', path)


signal.signal(signal.SIGTERM, save_before_exiting)

try:
    foo.run_training(batch_generator)
except KeyboardInterrupt:
    save_before_exiting()
    foo.shutdown()
    sys.exit(0)
