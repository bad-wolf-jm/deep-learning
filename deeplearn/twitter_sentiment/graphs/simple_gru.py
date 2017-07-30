import tensorflow as tf
from graphs.base.base_classifier import BaseClassifier


class SimpleGRUClassifier(BaseClassifier):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, seq_length=1024, hidden_states=128, embedding_dimension=512, num_classes=3, num_rnn_layers=3, **kwargs):
        super(SimpleGRUClassifier, self).__init__(seq_length=seq_length, embedding_dimension=embedding_dimension, num_classes=num_classes)
        self._input_dtype = tf.int32
        self.hidden_states = hidden_states
        self.number_of_layers = num_rnn_layers

    def init_model(self, trainable=False):
        x = tf.reshape(self._one_hot_input, [-1, self.seq_length, self.embedding_dimension])

        with tf.variable_scope('gru_encoder', reuse=None) as scope:
            encoding_layers = [tf.nn.rnn_cell.GRUCell(self.hidden_states) for _ in range(self.number_of_layers)]
            if self.number_of_layers > 1:
                multicell = tf.nn.rnn_cell.MultiRNNCell(encoding_layers, state_is_tuple=True)
            else:
                multicell = encoding_layers[0]
            Yr, H = tf.nn.dynamic_rnn(multicell, x, dtype=tf.float32)
            encoded_text = Yr
            encoded_text = tf.reshape(encoded_text, [-1, self.hidden_states * self.seq_length])
            print('Yr=', encoded_text.get_shape())

        with tf.variable_scope('dense_classifier', reuse=None) as scope:
            _weights = self.var(input_shape=[self.hidden_states * self.seq_length, self.seq_length],
                                name='final_weights',
                                scope=scope,
                                trainable=True)
            _biases = self.var(input_shape=[1, self.seq_length],
                               name='final_biases',
                               scope=scope,
                               trainable=True)
            _weights_2 = self.var(input_shape=[self.seq_length, self.num_classes],
                                  name='final_weights_2',
                                  scope=scope,
                                  trainable=True)
            _biases_2 = self.var(input_shape=[1, self.num_classes],
                                 name='final_biases_2',
                                 scope=scope,
                                 trainable=True)
            self.graph_output = tf.matmul(tf.nn.sigmoid(tf.matmul(encoded_text, _weights) + _biases), _weights_2) + _biases_2

    def build_inference_model(self, trainable=False):
        super(SimpleGRUClassifier, self).build_inference_model()
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)


if __name__ == '__main__':
    from graphs.base.test import test_graph_type
    test_graph_type(SimpleGRUClassifier)
    # test the graph
