import numpy as np
import tensorflow as tf
from models.categorical_encoder import CategoricalEncoder
from models.tf_session import tf_session
from models.base_model import BaseModel
import time


class SimpleGRUClassifier(BaseModel):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, seq_length=1024, hidden_states=128, embedding_dimension=512, num_classes=3, num_rnn_layers=3):
        super(SimpleGRUClassifier, self).__init__()
        self._input_dtype = tf.int32
        self.seq_length = seq_length
        self.hidden_states = hidden_states
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes
        self.number_of_layers = num_rnn_layers

    def init_model(self, trainable=False):
        with tf.variable_scope('embedding', reuse=None) as scope:
            self._input = tf.placeholder(dtype=self._input_dtype, shape=[None, self.seq_length], name="INPUT")
            x = tf.reshape(self._input, [-1])
            self._one_hot_input = tf.one_hot(x, depth=256, axis=1)
            x = tf.reshape(self._one_hot_input, [-1, 1024, 256])

        with tf.variable_scope('gru_encoder', reuse=None) as scope:
            encoding_layers = [tf.nn.rnn_cell.GRUCell(self.hidden_states) for _ in range(self.number_of_layers)]
            if self.number_of_layers > 1:
                multicell = tf.nn.rnn_cell.MultiRNNCell(encoding_layers, state_is_tuple=True)
            else:
                multicell = encoding_layers[0]
            Yr, H = tf.nn.dynamic_rnn(multicell, x, dtype=tf.float32)
            encoded_text = Yr  # [:, -1, :]  # get the last output
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
            self.output_predicted = tf.matmul(tf.nn.sigmoid(tf.matmul(encoded_text, _weights) + _biases), _weights_2) + _biases_2

    def build_inference_model(self, trainable=False):
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)

    def build_training_model(self):
        with tf.variable_scope('training', reuse=None) as scope:
            self.build_inference_model(trainable=True)
            self.output_expected = tf.placeholder(dtype=self._input_dtype, shape=[None, 1], name="INPUT")
            self.output_expected_oh = tf.one_hot(self.output_expected, depth=self.num_classes, axis=-1)
            self.output_expected_oh = tf.reshape(self.output_expected_oh, [-1, self.num_classes])
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_predicted, labels=self.output_expected_oh)
            self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
            self.predicted_value = tf.argmax(self.output_predicted, 1)
            self.true_value = tf.reshape(self.output_expected, [-1])
            self.batch_loss = tf.reduce_mean(loss, axis=0)
            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.true_value, tf.uint8), tf.cast(self.predicted_value, tf.uint8)), tf.float32))

    def train(self, train_x, train_y):
        t_1 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self._input: batch_x, self.output_expected: batch_y}
        _, lo, acc = tf_session().run([self.train_step, self.batch_loss, self.batch_accuracy], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': float(lo), 'accuracy': float(acc), 'time': batch_time}

    def validate(self, train_x, train_y):
        t_1 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self._input: batch_x, self.output_expected: batch_y}
        lo, acc = tf_session().run([self.batch_loss, self.batch_accuracy], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': float(lo), 'accuracy': float(acc), 'time': batch_time}

    def test(self, train_x, train_y):
        t_0 = time.time()
        batch_x = [self.pad(element, self.seq_length) for element in train_x]
        batch_y = [element for element in train_y]
        feed_dict = {self._input: batch_x, self.output_expected: batch_y}
        t_v, p_v, lo, acc, o_p = tf_session().run([self.true_value, self.predicted_value, self.batch_loss, self.batch_accuracy, tf.nn.softmax(self.output_predicted)], feed_dict=feed_dict)
        t = time.time() - t_0
        batch_strings = []
        for i, line in enumerate(batch_x):
            l = bytes([x for x in line if x != 0]).decode('utf8', 'ignore')
            batch_strings.append(l)
            x = l[:50]
            p = "." * (50 - len(x))
            print(x + p, o_p[i])
        return {'loss': lo, 'accuracy': acc, 'time': t, 'output': zip(batch_strings, t_v, p_v)}

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array


class SimpleGRUClassifierConv(SimpleGRUClassifier):
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
        self.convolutional_features = [32, 64, 128, 512]
        self.window_sizes = [3, 3, 3, 3]
        self.pooling_sizes = [5, 5, 5, 5]
        self.pooling_strides = [2, 2, 2, 2]
        with tf.variable_scope('embedding', reuse=None) as scope:
            self._input = tf.placeholder(dtype=self._input_dtype, shape=[None, self.seq_length], name="INPUT")
            x = tf.reshape(self._input, [-1])
            self._one_hot_input = tf.one_hot(x, depth=256, axis=1)
            x = tf.reshape(self._one_hot_input, [-1, 1024, 256])

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
            conv_1 = self.convolutional_block(i_tensor=encoded_text,
                                              i_features=1,
                                              o_features=self.convolutional_features[0],
                                              window_size=self.window_sizes[0],
                                              scope='layer_1')
            pool_1 = self.max_pool_layer(i_tensor=conv_1,
                                         window_size=self.pooling_sizes[0],
                                         strides=self.pooling_strides[0],
                                         scope='layer_1')
            conv_2 = self.convolutional_block(i_tensor=pool_1,
                                              i_features=self.convolutional_features[0],
                                              o_features=self.convolutional_features[1],
                                              window_size=self.window_sizes[1],
                                              scope='layer_2')
            pool_2 = self.max_pool_layer(i_tensor=conv_2,
                                         window_size=self.pooling_sizes[1],
                                         strides=self.pooling_strides[1],
                                         scope=scope)
            conv_3 = self.convolutional_block(i_tensor=pool_2,
                                              i_features=self.convolutional_features[1],
                                              o_features=self.convolutional_features[2],
                                              window_size=self.window_sizes[2],
                                              scope='layer_3')
            pool_3 = self.max_pool_layer(i_tensor=conv_3,
                                         window_size=self.pooling_sizes[2],
                                         strides=self.pooling_strides[2],
                                         scope=scope)
            conv_4 = self.convolutional_block(i_tensor=conv_3,
                                              i_features=self.convolutional_features[2],
                                              o_features=self.convolutional_features[3],
                                              window_size=self.window_sizes[3],
                                              scope='layer_4')
            pool_4 = self.max_pool_layer(i_tensor=conv_4,
                                         window_size=self.pooling_sizes[3],
                                         strides=self.pooling_strides[3],
                                         scope=scope)

            D = pool_4.shape[1].value * pool_4.shape[2].value * pool_4.shape[3].value
            pool_4 = tf.reshape(pool_4, [-1, D])

            _weights = self.var(input_shape=[D, self.num_classes],
                                name='final_weights',
                                scope=scope,
                                trainable=True)
            _biases = self.var(input_shape=[1, self.num_classes],
                               name='final_biases',
                               scope=scope,
                               trainable=True)
            self.output_predicted = tf.matmul(conv_4, _weights) + _biases

    def build_inference_model(self, trainable=False):
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)

    def build_training_model(self):
        with tf.variable_scope('training', reuse=None) as scope:
            self.build_inference_model(trainable=True)
            self.output_expected = tf.placeholder(dtype=self._input_dtype, shape=[None, 1], name="INPUT")
            self.output_expected_oh = tf.one_hot(self.output_expected, depth=self.num_classes, axis=-1)
            self.output_expected_oh = tf.reshape(self.output_expected_oh, [-1, self.num_classes])
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_predicted, labels=self.output_expected_oh)
            self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
            self.predicted_value = tf.argmax(self.output_predicted, 1)
            self.true_value = tf.reshape(self.output_expected, [-1])
            self.batch_loss = tf.reduce_mean(loss, axis=0)
            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.true_value, tf.uint8), tf.cast(self.predicted_value, tf.uint8)), tf.float32))
