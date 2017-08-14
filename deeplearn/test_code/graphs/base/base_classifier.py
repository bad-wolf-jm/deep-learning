import time
import numpy as np
import tensorflow as tf
from graphs.base.base_graph import BaseGraph


class BaseClassifier(BaseGraph):
    def __init__(self, seq_length=140,  embedding_dimension=256, num_classes=5):
        super(BaseClassifier, self).__init__()
        self.seq_length = seq_length
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes
        self.input = None
        self.true_values = None
        self.graph_output = None
        self.predicted_values = None

    def build_inference_model(self):
        with tf.variable_scope('input'):
            self.input = tf.placeholder('uint8', shape=[None, self.seq_length], name="INPUT")
            self._one_hot_input = tf.one_hot(self.input, depth=self.embedding_dimension, axis=-1)
            self._input_tensor = self._one_hot_input

    def build_training_model(self):
        self.build_inference_model()
        with tf.variable_scope('training'):
            with tf.variable_scope('output'):
                self.true_values = tf.placeholder(dtype=tf.uint8, shape=[None, 1], name="OUTPUT")
                self._one_hot_output = tf.one_hot(self.true_values, depth=self.num_classes, axis=-1)
            SXEWL = tf.nn.softmax_cross_entropy_with_logits
            self.loss = SXEWL(logits=self.graph_output, labels=self._one_hot_output)
            self.predicted_values = tf.cast(tf.argmax(self.graph_output, 1), tf.uint8)
            foo = tf.reshape(self.true_values, [-1])
            self.batch_loss = tf.reduce_mean(self.loss, axis=0)
            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(foo, self.predicted_values), tf.float32))


    def train_setup(self, optimizer, learning_rate, **kwargs):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.optimizer_args = kwargs
        self.train_step = self.optimizer(learning_rate=self.learning_rate, **self.optimizer_args)
        self.train_step = self.train_step.minimize(self.loss)

    def test(self, train_x, train_y, session=None):
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        t_0 = time.time()
        feed_dict = {self.input: batch_x, self.true_values: batch_y}
        p_v, lo, acc = self.run_ops(session, [self.predicted_values, self.batch_loss, self.batch_accuracy],
                                    feed_dict=feed_dict)
        batch_strings = []
        for line in batch_x:
            l = bytes([x for x in line if x != 0]).decode('utf8', 'ignore')
            batch_strings.append(l)
        t = time.time() - t_0
        d = {'loss': lo, 'accuracy': acc, 'time': t, 'output': zip(batch_strings, np.reshape(batch_y, [-1]), p_v)}
        return d

    def train(self, train_x, train_y, session=None):
        t_1 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self.input: batch_x, self.true_values: batch_y}
        _, b_loss, b_accuracy = self.run_ops(session,
                                             [self.train_step, self.batch_loss, self.batch_accuracy],
                                             feed_dict=feed_dict)
        batch_time = time.time() - t_1
        d = {'loss': b_loss, 'accuracy': b_accuracy, 'time': batch_time, 'output': []}
        return d

    def validate(self, train_x, train_y, session=None):
        t_1 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self.input: batch_x, self.true_values: batch_y}
        loss, accuracy = self.run_ops(session, [self.batch_loss, self.batch_accuracy], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        d = {'loss': loss, 'accuracy': accuracy,  'time': batch_time, 'output': []}
        return d

    def pad(self, array, length, padding_value=0):
        array = list(array[:length])
        array += [padding_value] * (length - len(array))
        return array

    def half_learning_rate(self):
        self.train_setup(self.optimizer, self.learning_rate / 2, **self.optimizer_args)
