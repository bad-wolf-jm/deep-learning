import tensorflow as tf
from graphs.base.base_classifier import BaseClassifier


class SimpleGRUClassifierConv(BaseClassifier):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self,
                 seq_length=1024,
                 hidden_states=128,
                 embedding_dimension=512,
                 num_classes=3,
                 num_rnn_layers=3,
                 convolutional_features = [32, 64, 128, 512],
                 window_sizes = [3, 3, 3, 3],
                 pooling_sizes = [5, 5, 5, 5],
                 pooling_strides = [2, 2, 2, 2], **kwargs):
        super(SimpleGRUClassifierConv, self).__init__(seq_length=seq_length, embedding_dimension=embedding_dimension, num_classes=num_classes)
        self._input_dtype = tf.int32
        self.hidden_states = hidden_states
        self.number_of_layers = num_rnn_layers
        self.convolutional_features = convolutional_features
        self.window_sizes = window_sizes
        self.pooling_sizes = pooling_sizes
        self.pooling_strides = pooling_strides

    def convolutional_block(self, i_tensor, i_features, o_features, window_size=3, scope=None):
        with tf.variable_scope(scope):
            kernel = self.var(input_shape=[window_size, window_size, i_features, o_features], name='layer_1_convolution')
            x = tf.nn.conv2d(i_tensor, kernel, strides=[1, 1, 1, 1], padding="SAME")
            conv = tf.nn.relu(x)
            return conv

    def max_pool_layer(self, i_tensor, window_size, strides, scope=None):
        conv_block_1 = tf.reshape(i_tensor, [-1, i_tensor.shape[1].value, i_tensor.shape[2].value, i_tensor.shape[3].value])
        conv_block_1 = tf.nn.max_pool(conv_block_1, [1, window_size, window_size, 1], [1, strides, strides, 1], 'SAME')
        print(conv_block_1.get_shape())
        return conv_block_1

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
            encoded_text = tf.reshape(encoded_text, [-1, self.hidden_states, self.seq_length, 1])
            print('Yr=', encoded_text.get_shape())
        with tf.variable_scope('convolutional_classifier', reuse=None) as scope:
            i_features = 1
            conv_input = encoded_text
            for layer_index in range(len(self.convolutional_features)-1):
                conv_layer = self.convolutional_block(i_tensor=conv_input,
                                                  i_features=i_features,
                                                  o_features=self.convolutional_features[layer_index],
                                                  window_size=self.window_sizes[layer_index],
                                                  scope='layer_{}'.format(layer_index))
                pool_layer = self.max_pool_layer(i_tensor=conv_layer,
                                             window_size=self.pooling_sizes[layer_index],
                                             strides=self.pooling_strides[layer_index],
                                             scope='layer_{}'.format(layer_index))
                conv_input = pool_layer
                i_features = self.convolutional_features[layer_index]
            D = pool_layer.shape[1].value * pool_layer.shape[2].value * pool_layer.shape[3].value
            pool_layer = tf.reshape(pool_layer, [-1, D])
            _weights = self.var(input_shape=[D, self.num_classes],
                                name='final_weights',
                                scope=scope,
                                trainable=True)
            _biases = self.var(input_shape=[1, self.num_classes],
                               name='final_biases',
                               scope=scope,
                               trainable=True)
            self.graph_output = tf.matmul(pool_layer, _weights) + _biases

    def build_inference_model(self, trainable=False):
        super(SimpleGRUClassifierConv, self).build_inference_model()
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)


if __name__ == '__main__':
    from graphs.base.test import test_graph_type
    test_graph_type(SimpleGRUClassifierConv)
    #test the graph
