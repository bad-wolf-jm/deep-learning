import time
import tensorflow as tf
from graphs.base.base_classifier import BaseClassifier
from graphs.base.tf_session import tf_session

class ByteCNN(BaseClassifier):
    def __init__(self, seq_length=140, embedding_dimension=256, num_classes=5, level_features=[64, 64, 128, 256, 512],
                 sub_levels=[2, 2, 2, 2], classifier_layers=[4096, 2048, 2048]):
        super(ByteCNN, self).__init__(seq_length=seq_length, embedding_dimension=embedding_dimension, num_classes=num_classes)
        assert len(sub_levels) == len(level_features) - 1
        #self.input_width = seq_length
        #self.seq_length = seq_length
        #self.input_depth = input_depth
        #self.num_classes = num_classes
        self.level_features = level_features
        self.sub_levels = sub_levels
        self.classifier_layers = classifier_layers
        self.conv_block = {}
        self.conv_output = None

    def build_inference_model(self):
        super(ByteCNN, self).build_inference_model()
        with tf.variable_scope('convolutional_layer_1'):
            kernel = self.var(input_shape=[5, self.embedding_dimension, self.level_features[0]], name='layer_1_convolution')
            self.conv_block[0] = tf.nn.conv1d(self._input_tensor, kernel, stride=1, padding="SAME")

        for level, num_features in enumerate(self.level_features[:-1]):
            with tf.variable_scope('convolutional_block_' + str(level + 1)):
                c_b = self.convolutional_block(self.conv_block[level],
                                               self.level_features[level],
                                               self.level_features[level + 1],
                                               scope='L1')
                for sub_level in range(self.sub_levels[level] - 1):
                    c_b = self.convolutional_block(c_b,
                                                   self.level_features[level + 1],
                                                   self.level_features[level + 1],
                                                   scope='L' + str(sub_level + 2))
                m_p = self.max_pool_layer(c_b, 2, 2)
                self.conv_block[level + 1] = m_p
            self.conv_output = m_p

        self.conv_output = tf.reshape(self.conv_output, [-1, self.conv_output.shape[1].value * self.level_features[-1]])
        FC = tf.contrib.layers.fully_connected
        with tf.variable_scope("fc_classifier") as scope:
            x = FC(self.conv_output, self.classifier_layers[0], activation_fn=None)
            for num_output in self.classifier_layers[1:]:
                x = FC(x, num_output, activation_fn=None)

        self.decision_output = self.decision_layer_3 = FC(x, self.num_classes, activation_fn=None)
        self.graph_output = self.decision_output
        print(self.decision_layer_3.get_shape())

    def temporal_batch_normalize(self, i_tensor):
        mean, variance = tf.nn.moments(i_tensor, axes=[0, 1])
        o_tensor = tf.nn.batch_normalization(
            i_tensor, mean, variance, offset=None, scale=None, variance_epsilon=1e-08)
        return o_tensor

    def convolutional_block(self, i_tensor, i_features, o_features, scope=None):
        with tf.variable_scope(scope):
            kernel = self.var(input_shape=[3, i_features, o_features], name='layer_1_convolution')
            x = tf.nn.conv1d(i_tensor, kernel, stride=1, padding="SAME")
            x = self.temporal_batch_normalize(x)
            conv = tf.nn.relu(x)
            kernel = self.var(input_shape=[3, o_features, o_features], name='layer_2_convolution')
            x = tf.nn.conv1d(conv, kernel, stride=1, padding="SAME")
            x = self.temporal_batch_normalize(x)
            conv_block = tf.nn.relu(x)
            return conv_block

    def max_pool_layer(self, i_tensor, window_size, strides):
        conv_block_1 = tf.reshape(
            i_tensor, [-1, 1, i_tensor.shape[1].value, i_tensor.shape[2].value])
        conv_block_1 = tf.nn.max_pool(conv_block_1, [1, 1, window_size, 1], [1, 1, strides, 1], 'SAME')
        conv_block_1 = tf.reshape(
            conv_block_1, [-1, conv_block_1.shape[2].value, conv_block_1.shape[3].value])
        return conv_block_1


if __name__ == '__main__':
    from graphs.base.test import test_graph_type
    test_graph_type(ByteCNN)
    # test the graph
