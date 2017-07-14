#from convolutional_model_1 import model
#from models.byte_cnn.byte_cnn import ByteCNN
from models.tf_session import tf_session
import tensorflow as tf
import numpy as np
import sys
import signal
import os
from stream.receiver import DataReceiver
from train.summary import TrainingSummary


class TrainingSupervisor(object):
    def __init__(self, model):
        super(TrainingSupervisor, self).__init__()
        self.model = model
        self.batch_index = 0
        self.checkpoint_interval = 250
        self.checkpoint_keep = 10
        _r = os.path.expanduser('~/.sentiment_analysis/checkpoints/')
        _r = os.path.join(_r, type(model).__name__)
        self.checkpoint_root = _r
        if not os.path.exists(self.checkpoint_root):
            os.makedirs(self.checkpoint_root)
        self.summary = TrainingSummary(summary_span=None,
                                       fields=['loss', 'accuracy', 'time'])

    def save_model_as(self, file_name):
        saver = tf.train.Saver()
        checkpoint_name = file_name
        x = os.path.join(self.checkpoint_root, checkpoint_name)
        path = saver.save(tf_session(), x)
        return path

    def save_model_image(self, *a):
        return self.save_model_as('restore-model-image.chkpt')

    def train_on_batch(self, train_x, train_y):
        self.batch_index += 1
        d = self.train_step(train_x, train_y)
        d = {'accuracy': float(d['accuracy']),
             'loss': float(d['loss']),
             'time': float(d['time'])}
        self.summary.add_to_summary('train', self.batch_index, **d)
        if (self.batch_index % self.checkpoint_interval) == 0:
            path = self.save_model_as('training-checkpoint.chkpt')
            print('Saving checkpoint:', path)
        return d

    def validate_on_batch(self, train_x, train_y):
        d = self.validation_step(train_x, train_y)
        d = {'accuracy': float(d['accuracy']),
             'loss': float(d['loss']),
             'time': float(d['time'])}
        self.summary.add_to_summary('validation', self.batch_index, **d)
        return d

    def get_train_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.summary.get_summary('train',
                                     min_batch_index=min_batch_index,
                                     max_batch_index=max_batch_index)
        return x

    def get_validation_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.summary.get_summary('validation',
                                     min_batch_index=min_batch_index,
                                     max_batch_index=max_batch_index)
        return x

    def run_training(self, restore_from_checkpoint = None):
        self.train_data_stream = DataReceiver(bind='*')
        self.train_command_server = DataReceiver(port=99887)
        tf_session().run(tf.global_variables_initializer())
        if restore_from_checkpoint is not None:
            _p = os.path.join(self.checkpoint_root, restore_from_checkpoint+'.chkpt')
            if os.path.exists(_p):
                saver = tf.train.Saver()
                saver.restore(tf_session(), _p)
        #else:

        self.train_data_stream.register_action_handler('train', self.train_on_batch)
        self.train_data_stream.register_action_handler('validate', self.validate_on_batch)
        self.train_command_server.register_action_handler('get_train_batch_data', self.get_train_summary)
        self.train_command_server.register_action_handler('get_validation_batch_data', self.get_validation_summary)
        self.train_command_server.start(True)
        self.train_data_stream.start(False)

    def shutdown(self):
        self.train_data_stream.shutdown()
        self.train_command_server.shutdown()
