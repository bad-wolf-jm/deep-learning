#from convolutional_model_1 import model
#from models.byte_cnn.byte_cnn import ByteCNN
from models.tf_session import tf_session
import tensorflow as tf
import numpy as np
import sys
import signal
from stream.receiver import DataReceiver
from train.summary import TrainingSummary


class TrainingSupervisor(object):
    def __init__(self, model):
        super(TrainingSupervisor, self).__init__()
        self.model = model
        self.batch_index = 0
        self.summary = TrainingSummary(summary_span=None,
                                       fields=['loss', 'accuracy', 'time'])

    def train_on_batch(self, train_x, train_y):
        self.batch_index += 1
        d = self.train_step(train_x, train_y)
        d = {'accuracy': float(d['accuracy']),
             'loss': float(d['loss']),
             'time': float(d['time'])}
        self.summary.add_to_summary('train', self.batch_index, **d)
        return d

    def validate_on_batch(self, train_x, train_y):
        d = self.validation_step(train_x, train_y)
        d = {'accuracy': float(d['accuracy']),
             'loss': float(d['loss']),
             'time': float(d['time'])}
        self.summary.add_to_summary('validation', self.batch_index, **d)
        return d

    def get_train_summary(self, min_batch_index=None, max_batch_index=None):
        return self.get_summary('train', min_batch_index, max_batch_index)

    def get_validation_summary(self, min_batch_index=None, max_batch_index=None):
        return self.get_summary('validation', min_batch_index, max_batch_index)

    def run_training(self):
        with tf_session() as session:
            foo = DataReceiver()
            bar = DataReceiver(port=99887)
            session.run(tf.global_variables_initializer())
            foo.register_action_handler('train', self.train_on_batch)
            foo.register_action_handler('validate', self.validate_on_batch)
            bar.register_action_handler('get_train_batch_data', self.get_train_summary)
            bar.register_action_handler('get_validation_batch_data', self.get_validation_summary)
            bar.start(True)
            foo.start(False)
