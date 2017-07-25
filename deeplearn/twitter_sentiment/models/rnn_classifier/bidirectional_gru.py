import numpy as np
import tensorflow as tf
from models.categorical_encoder import CategoricalEncoder
from models.tf_session import tf_session
from models.base_model import BaseModel
import time


class Tweet2Vec_BiGRU(BaseModel):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, seq_length=1024, hidden_states=128, embedding_dimension=512, num_classes=3):
        super(Tweet2Vec_BiGRU, self).__init__()
        self._input_dtype = tf.int32
        self.seq_length = seq_length
        self.hidden_states = hidden_states
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes

    def init_model(self, trainable=False):
        with tf.variable_scope('embedding', reuse=None) as scope:
            self._input = tf.placeholder(dtype=self._input_dtype, shape=[None, self.seq_length], name="INPUT")
            x = tf.reshape(self._input, [-1])
            self._one_hot_input = tf.one_hot(x, depth=256, axis=1)
            x = tf.reshape(self._one_hot_input, [-1, 1024, 256])

        with tf.variable_scope('bidirectional_encoder', reuse=None) as scope:
            self.forward_gru = tf.nn.rnn_cell.GRUCell(self.hidden_states, activation=None)
            self.backward_gru = tf.nn.rnn_cell.GRUCell(self.hidden_states, activation=None)
            output, output_states = tf.nn.bidirectional_dynamic_rnn(self.forward_gru, self.backward_gru, x, dtype=tf.float32)
            self.fw_gru_output = tf.reshape(output[0][:, -1:], [-1, self.hidden_states])
            self.bk_gru_output = tf.reshape(output[1][:, :1], [-1, self.hidden_states])
            self.fw_gru_output_state = output_states[0]
            self.bk_gru_output_state = output_states[1]

        with tf.variable_scope('encoder_linear_combination', reuse=None) as scope:
            self.foward_weights = self.var(input_shape=[self.hidden_states, self.embedding_dimension], name='fw_weights', scope=scope, trainable=trainable)
            self.backward_weights = self.var(input_shape=[self.hidden_states, self.embedding_dimension], name='bk_weights', scope=scope, trainable=trainable)
        with tf.variable_scope('output', reuse=None) as scope:
            self.enc_output = tf.add(tf.matmul(self.bk_gru_output, self.backward_weights),
                                     tf.matmul(self.fw_gru_output, self.foward_weights))

    def build_inference_model(self, trainable=False):
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)

    def build_training_model(self):
        with tf.variable_scope('training', reuse=None) as scope:
            self.build_inference_model(trainable=True)
            self.final_layer_weights = self.var(input_shape=[self.embedding_dimension, self.num_classes],
                                                name='final_weights',
                                                scope=scope,
                                                trainable=True)
            self.output_predicted = tf.matmul(self.enc_output, self.final_layer_weights)
            self.output_expected = tf.placeholder(dtype=self._input_dtype, shape=[None, 1], name="INPUT")
            self.output_expected_oh = tf.one_hot(self.output_expected, depth=self.num_classes, axis=-1)
            self.output_expected_oh = tf.reshape(self.output_expected_oh, [-1, self.num_classes])
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_predicted, labels=self.output_expected_oh)
            self.train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
            self.predicted_value = tf.argmax(self.output_predicted, 1)
            self.true_value = tf.reshape(self.output_expected, [-1])
            self.batch_loss = tf.reduce_mean(loss, axis=0)
            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.true_value, tf.uint8), tf.cast(self.predicted_value, tf.uint8)), tf.float32))

    def train(self, train_x, train_y, session=None):
        t_1 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self._input: batch_x, self.output_expected: batch_y}
        _, lo, acc = self.run_ops(session,
                                  [self.train_step, self.batch_loss, self.batch_accuracy],
                                  feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': float(lo), 'accuracy': float(acc), 'time': batch_time}

    def validate(self,  train_x, train_y, session=None):
        t_1 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self._input: batch_x, self.output_expected: batch_y}
        lo, acc = self.run_ops(session,
                               [self.batch_loss, self.batch_accuracy],
                               feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': float(lo), 'accuracy': float(acc), 'time': batch_time}

    def test(self,  train_x, train_y, session=None):
        t_0 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self._input: batch_x, self.output_expected: batch_y}
        t_v, p_v, lo, acc, o_p, o_e = self.run_ops(session,
                                                   [self.true_value, self.predicted_value, self.batch_loss, self.batch_accuracy,
                                                    tf.nn.softmax(self.output_predicted), self.output_expected_oh],
                                                   feed_dict=feed_dict)
        t = time.time() - t_0
        batch_strings = []
        for i, line in enumerate(batch_x):
            l = bytes([x for x in line if x != 0]).decode('utf8', 'ignore')
            batch_strings.append(l)
            x = l[:50]
            p = "." * (50 - len(x))
            print(x + p, o_p[i],  o_e[i])
        return {'loss': lo, 'accuracy': acc, 'time': t, 'output': zip(batch_strings, t_v, p_v)}

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array
