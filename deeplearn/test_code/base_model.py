import numpy as np
import tensorflow as tf
import time
import os
import sys
from models.tf_session import tf_session
import pickle


class StopTraining(Exception):
    pass

class BaseModel(object):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """

    def __init__(self, input_dimension=256, output_dimension=8):
        super(BaseModel, self).__init__()
        self._variables = {}
        self.batch_loss = None
        self.batch_accuracy = None
        #self.graph = tf.Graph()

    def build_inference_model(self):
        raise NotImplementedError()

    def build_training_model(self):
        raise NotImplementedError()

    def var(self, input_shape, trainable=True, name="variable", scope=None):
        full_variable_name = '{scope}/{name}'.format(scope=tf.get_variable_scope().name, name=name)
        initializer = self._variables.get(full_variable_name, None)
        if initializer is None:
            initializer = tf.random_normal(input_shape, stddev=0.35)
        v = tf.Variable(initializer, name=name, trainable=trainable)
        self._variables[full_variable_name] = v
        return v

    def initialize(self):
        op = tf.variables_initializer(self._variables.values())
        tf_session().run(op)

    def train(self, batch_x, batch_y, session=None):
        t_1 = time.time()
        feed_dict = {self._input: batch_x, self._output: batch_y}
        _, lo, acc = self.run_ops(session,
                                  [self.train_step, self.batch_loss, self.batch_accuracy],
                                  feed_dict=feed_dict)

        batch_time = time.time() - t_1
        return {'loss': lo, 'accuracy': acc, 'time': batch_time}

    def validate(self, batch_x, batch_y, session=None):
        t_1 = time.time()
        feed_dict = {self._input: batch_x, self._output: batch_y}
        lo, acc = self.run_ops(session,
                               [self.batch_loss, self.batch_accuracy],
                               feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': lo, 'accuracy': acc,  'time': batch_time}


    def run_ops(self, session = None, *args, **kwargs):
        session = session or tf_session()
        return session.run(*args, **kwargs)
