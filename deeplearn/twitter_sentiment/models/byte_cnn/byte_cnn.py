#import numpy as np
import time
import tensorflow as tf
#import numpy as np
from models.categorical_encoder import CategoricalEncoder
from models.base_model import BaseModel
from models.tf_session import tf_session

#CNN_LEVEL_0_FEATURES = 64
#CNN_LEVEL_1_FEATURES = 64
#CNN_LEVEL_2_FEATURES = 128
#CNN_LEVEL_3_FEATURES = 256
#CNN_LEVEL_4_FEATURES = 512


class ByteCNN(BaseModel):
    def __init__(self, seq_length=140, input_depth=256, num_categories=5, level_features=[64, 64, 128, 256, 512],
                 sub_levels=[2, 2, 2, 2], classifier_layers=[4096, 2048, 2048]):
        super(ByteCNN, self).__init__()
        assert len(sub_levels) == len(level_features) - 1
        self.input_width = seq_length
        self.seq_length = seq_length
        self.input_depth = input_depth
        self.num_categories = num_categories
        self.level_features = level_features
        self.sub_levels = sub_levels
        self.classifier_layers = classifier_layers
        self.conv_block = {}
        self.conv_output = None

    def build_inference_model(self):
        with tf.variable_scope('input'):
            self._input = tf.placeholder('uint8', shape=[None, self.input_width], name="INPUT")
            self._one_hot_input = tf.one_hot(self._input, depth=256, axis=-1)
            self._input_tensor = self._one_hot_input

        with tf.variable_scope('convolutional_layer_1'):
            kernel = self.var(input_shape=[5, self.input_depth, self.level_features[0]], name='layer_1_convolution')
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

        self.decision_output = self.decision_layer_3 = FC(x, self.num_categories, activation_fn=None)
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
                self._one_hot_output = tf.one_hot(self._output, depth=self.num_categories, axis=-1)
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

    def test(self, train_x, train_y):
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        t_0 = time.time()
        feed_dict = {self._input: batch_x, self._output: batch_y}
        t_v, p_v, lo, acc = tf_session().run([self.true_value, self.predicted_value, self.batch_loss, self.batch_accuracy], feed_dict=feed_dict)
        t = time.time() - t_0
        batch_strings = []
        for line in batch_x:
            l = bytes([x for x in line if x != 0]).decode('utf8', 'ignore')
            batch_strings.append(l)
        return {'loss': lo, 'accuracy': acc, 'time': t, 'output': zip(batch_strings, t_v, p_v)}

    def train(self, train_x, train_y):
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        d = super(ByteCNN, self).train(batch_x, batch_y)
        print (d)
        return d

    def validate(self, train_x, train_y):
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        d = super(ByteCNN, self).validate(batch_x, batch_y)
        #d = self.model.validate(batch_x, batch_y)
        print (d)
        return d

    #def test_model(self, train_x, train_y):
    #    batch_x = np.array([self.pad(element, self.seq_length) for element in train_x])
    #    batch_y = np.array([element for element in train_y])
    #    d = super(ByteCNN, self).validate(batch_x, batch_y)
    #    d = self.model.test(batch_x, batch_y)
    #    return d

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array
