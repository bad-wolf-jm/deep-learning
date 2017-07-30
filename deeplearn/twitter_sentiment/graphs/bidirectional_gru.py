import numpy as np
import tensorflow as tf
#from models.categorical_encoder import CategoricalEncoder
from graphs.base.tf_session import tf_session
from graphs.base.base_classifier import BaseClassifier
import time


class Tweet2Vec_BiGRU(BaseClassifier):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, seq_length=1024, hidden_states=128, embedding_dimension=512, num_classes=3, number_of_layers=1, **kwargs):
        super(Tweet2Vec_BiGRU, self).__init__(seq_length=seq_length, embedding_dimension=embedding_dimension, num_classes=num_classes)
        self._input_dtype = tf.int32
        self.hidden_states = hidden_states
        self.number_of_layers = number_of_layers

    def init_model(self, trainable=False):
        x = tf.reshape(self._one_hot_input, [-1, self.seq_length, self.embedding_dimension])

        with tf.variable_scope('bidirectional_encoder', reuse=None) as scope:
            self.forward_gru = self.multi_layer_rnn()
            self.backward_gru = self.multi_layer_rnn()
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
            self.final_layer_weights = self.var(input_shape=[self.embedding_dimension, self.num_classes],
                                                name='final_weights',
                                                scope=scope,
                                                trainable=True)
            self.graph_output = tf.matmul(self.enc_output, self.final_layer_weights)

    def multi_layer_rnn(self):
        layers = [tf.nn.rnn_cell.GRUCell(self.hidden_states) for _ in range(self.number_of_layers)]
        if self.number_of_layers > 1:
            return tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)
        else:
            return layers[0]

    def build_inference_model(self, trainable=False):
        super(Tweet2Vec_BiGRU, self).build_inference_model()
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)


if __name__ == '__main__':
    from graphs.base.test import test_graph_type
    test_graph_type(Tweet2Vec_BiGRU)
    # test the graph
