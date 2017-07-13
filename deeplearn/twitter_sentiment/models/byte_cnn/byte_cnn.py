#import numpy as np
import tensorflow as tf
from models.categorical_encoder import CategoricalEncoder
from models.base_model import BaseModel
from models.tf_session import tf_session

CNN_LEVEL_0_FEATURES = 64
CNN_LEVEL_1_FEATURES = 64
CNN_LEVEL_2_FEATURES = 128
CNN_LEVEL_3_FEATURES = 256
CNN_LEVEL_4_FEATURES = 512


class ByteCNN(BaseModel):
    def __init__(self):
        super(ByteCNN, self).__init__()
        self.input_width = 140
        self.input_depth = 256
        self.encoded_input_depth = 8
        self.categories = 3
        self.byte_encoder = CategoricalEncoder.instance_from_pickle('weights.pkl')
        self._variables = {}

    def build_inference_model(self):
        with tf.variable_scope('input'):
            self._input = tf.placeholder('uint8', shape=[None, self.input_width], name="INPUT")
            self._one_hot_input = tf.one_hot(self._input, depth=256, axis=-1)
            stacked_one_hot = tf.reshape(self._one_hot_input, [-1, self.input_depth])
            encoded_batch = tf.matmul(stacked_one_hot, self.byte_encoder._encode_weights)
            #!NOTE input_tensor has shape [batch, input_width, num_input_channels], 8 channels
            self._input_tensor = tf.reshape(encoded_batch, [-1, self.input_width, self.encoded_input_depth])

        with tf.variable_scope('convolutional_layer_1'):
            kernel = self.var(input_shape=[5, self.encoded_input_depth, CNN_LEVEL_0_FEATURES], name='layer_1_convolution')
            #!NOTE: conv bas a shape of [batch, input_width, num_input_channels]
            self.conv = tf.nn.conv1d(self._input_tensor, kernel, stride=1, padding="SAME")

        with tf.variable_scope('convolutional_block_1'):
            c_b = self.convolutional_block(self.conv, CNN_LEVEL_0_FEATURES, CNN_LEVEL_1_FEATURES, scope='L1')
            c_b = self.convolutional_block(c_b, CNN_LEVEL_1_FEATURES, CNN_LEVEL_1_FEATURES, scope='L2')
            m_p = self.max_pool_layer(c_b, 2, 2)
            self.conv_block_1 = m_p

        with tf.variable_scope('convolutional_block_2'):
            c_b = self.convolutional_block(self.conv_block_1, CNN_LEVEL_1_FEATURES, CNN_LEVEL_2_FEATURES, scope='L1')
            c_b = self.convolutional_block(c_b, CNN_LEVEL_2_FEATURES, CNN_LEVEL_2_FEATURES, scope='L2')
            m_p = self.max_pool_layer(c_b, 2, 2)
            self.conv_block_2 = m_p

        with tf.variable_scope('convolutional_block_3'):
            c_b = self.convolutional_block(self.conv_block_2, CNN_LEVEL_2_FEATURES, CNN_LEVEL_3_FEATURES, scope='L1')
            c_b = self.convolutional_block(c_b, CNN_LEVEL_3_FEATURES, CNN_LEVEL_3_FEATURES, scope='L2')
            m_p = self.max_pool_layer(c_b, 2, 2)
            self.conv_block_3 = m_p

        with tf.variable_scope('convolutional_block_4'):
            c_b = self.convolutional_block(self.conv_block_3, CNN_LEVEL_3_FEATURES, CNN_LEVEL_4_FEATURES, scope='L1')
            c_b = self.convolutional_block(c_b, CNN_LEVEL_4_FEATURES, CNN_LEVEL_4_FEATURES, scope='L2')
            m_p = self.max_pool_layer(c_b, 2, 2)
            self.conv_block_4 = m_p

        with tf.variable_scope('fully_connected_layer'):
            weights = self.var(input_shape=[self.conv_block_4.shape[1].value * CNN_LEVEL_4_FEATURES, 2048], name='projection_1')
            self.convx = tf.reshape(self.conv_block_4, [-1, self.conv_block_4.shape[1].value * CNN_LEVEL_4_FEATURES])
            #NOTE [batch_size, 2048]
            self.decision_layer_1 = tf.nn.relu(tf.matmul(self.convx, weights))

        with tf.variable_scope('fully_connected_layer_2'):
            weights = self.var(input_shape=[2048, 2048], name='projection_1')
            #NOTE [batch_size, 2]
            self.decision_layer_2 = tf.nn.relu(tf.matmul(self.decision_layer_1, weights))

        with tf.variable_scope('fully_connected_layer_3'):
            weights = self.var(input_shape=[2048, self.categories], name='projection_1')
            #NOTE [batch_size, 2]
            self.decision_layer_3 = tf.matmul(self.decision_layer_2, weights)

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

    def var(self, input_shape, trainable=True, name="variable", scope=None):
        full_variable_name = '{scope}/{name}'.format(scope=tf.get_variable_scope().name, name=name)
        initializer = self._variables.get(full_variable_name, None)
        if initializer is None:
            initializer = tf.random_normal(input_shape, stddev=0.35)
        v = tf.Variable(initializer, name=name)
        self._variables[full_variable_name] = v
        return v

    def initialize(self):
        op = tf.variables_initializer(self._variables.values())
        tf_session().run(op)

    def build_training_model(self):
        self.build_inference_model()
        with tf.variable_scope('training_ops'):
            with tf.variable_scope('output'):
                self._output = tf.placeholder(dtype=tf.uint8, shape=[None, 1], name="OUTPUT")
                self._one_hot_output = tf.one_hot(self._output, depth=self.categories, axis=-1)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.decision_layer_3, labels=self._one_hot_output)
            self.train_step = tf.train.MomentumOptimizer(learning_rate=0.000001, momentum=0.9).minimize(loss)
            self.predicted_value = tf.argmax(self.decision_layer_3, 1)
            self.true_value = tf.reshape(self._output, [-1])
            self.batch_loss = tf.reduce_mean(loss, axis=0)
            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.true_value, tf.cast(self.predicted_value, tf.uint8)), tf.float32))

    def debug_validate(self, batch_x, batch_y):
        feed_dict = {self._input: batch_x, self._output: batch_y}
        t_v, p_v, lo, acc = tf_session().run([self.true_value, self.predicted_value, self.batch_loss, self.batch_accuracy], feed_dict=feed_dict)
        print()
        print(t_v)
        print(p_v)
        return {'loss': lo, 'accuracy': acc}

    def test(self, batch_x, batch_y):
        feed_dict = {self._input: batch_x, self._output: batch_y}
        t_v, p_v, lo, acc = tf_session().run([self.true_value, self.predicted_value, self.batch_loss, self.batch_accuracy], feed_dict=feed_dict)
        batch_strings = []
        for line in batch_x:
            l = bytes([x for x in line if x != 0]).decode('utf8', 'ignore')
            batch_strings.append(l)
        return {'loss': lo, 'accuracy': acc, 'time':0, 'output':zip(batch_strings, t_v, p_v)}
