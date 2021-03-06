import numpy as np
import tensorflow as tf
from models.categorical_encoder import CategoricalEncoder
from models.tf_session import tf_session
from models.base_model import BaseModel
import time
import string

EMBEDDING_DIMENSION = 6
NUM_FILTERS = 512


class Tweet2Vec_LSTM(BaseModel):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, seq_length=1024, hidden_states=128, embedding_dimension=64, num_classes=3):
        super(Tweet2Vec_LSTM, self).__init__()
        self._input_dtype = tf.int32
        self.max_message_length = 16

        self.byte_encoding_depth = 256
        self.convolutional_features = 256
        self.encoder_internal_size = 32
        self.decoder_internal_size = 32
        self.num_decoder_layers = 2
        self.pooling_sizes = [3, 3]
        self.pooling_strides = [2, 2]
        self.window_sizes = [7, 7, 3, 3]
        self.dropout_keep_probability = 0.7

    def init_model(self, trainable=False):
        with tf.variable_scope('input', reuse=None) as scope:
            self.input_layer = tf.placeholder('int32', shape=[None, self.max_message_length], name="INPUT")
            input_layer = tf.one_hot(self.input_layer, self.byte_encoding_depth)

        with tf.variable_scope('encoder', reuse=None) as scope:
            with tf.variable_scope('convolution', reuse=None) as scope:
                conv_1 = self.convolutional_block(i_tensor=input_layer,
                                                  i_features=self.byte_encoding_depth,
                                                  o_features=self.convolutional_features,
                                                  window_size=self.window_sizes[0],
                                                  scope='layer_1')
                pool_1 = self.max_pool_layer(i_tensor=conv_1,
                                             window_size=self.pooling_sizes[0],
                                             strides=self.pooling_strides[0],
                                             scope='layer_1')
                conv_2 = self.convolutional_block(i_tensor=pool_1,
                                                  i_features=self.convolutional_features,
                                                  o_features=self.convolutional_features,
                                                  window_size=self.window_sizes[1],
                                                  scope='layer_2')
                pool_2 = self.max_pool_layer(i_tensor=conv_2,
                                             window_size=self.pooling_sizes[1],
                                             strides=self.pooling_strides[1],
                                             scope=scope)
                conv_3 = self.convolutional_block(i_tensor=pool_2,
                                                  i_features=self.convolutional_features,
                                                  o_features=self.convolutional_features,
                                                  window_size=self.window_sizes[2],
                                                  scope='layer_3')
                conv_4 = self.convolutional_block(i_tensor=conv_3,
                                                  i_features=self.convolutional_features,
                                                  o_features=self.convolutional_features,
                                                  window_size=self.window_sizes[3],
                                                  scope='layer_4')

            with tf.variable_scope('lstm_encoder', reuse=None) as scope:
                encoding_layer = tf.nn.rnn_cell.LSTMCell(self.encoder_internal_size)
                Yr, H = tf.nn.dynamic_rnn(encoding_layer, conv_4, dtype=tf.float32)
                encoded_text = Yr[:, -1, :]  # get the last output
                print('Yr=', encoded_text.get_shape())

        with tf.variable_scope('decoder', reuse=None) as scope:
            encoded_text = tf.concat([encoded_text] * self.max_message_length, axis=1)
            encoded_text = tf.reshape(encoded_text, [-1, self.max_message_length, self.decoder_internal_size])
            cells = [tf.nn.rnn_cell.LSTMCell(self.decoder_internal_size, state_is_tuple=True) for _ in range(self.num_decoder_layers)]
            dropcells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_keep_probability) for cell in cells]
            multicell = tf.nn.rnn_cell.MultiRNNCell(dropcells, state_is_tuple=True)
            multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, output_keep_prob=self.dropout_keep_probability)
            Yr, H = tf.nn.dynamic_rnn(multicell, encoded_text, dtype=tf.float32)
            Yr = tf.reshape(Yr, [-1, self.decoder_internal_size])
            self.final_layer_weights = self.var(input_shape=[self.decoder_internal_size, self.byte_encoding_depth], name='layer_1_convolution')
            final_layer = tf.matmul(Yr, self.final_layer_weights)
            self.output_predicted = final_layer

    def convolutional_block(self, i_tensor, i_features, o_features, window_size=3, scope=None):
        with tf.variable_scope(scope):
            kernel = self.var(input_shape=[window_size, i_features, o_features], name='layer_1_convolution')
            x = tf.nn.conv1d(i_tensor, kernel, stride=1, padding="SAME")
            conv = tf.nn.relu(x)
            return conv

    def max_pool_layer(self, i_tensor, window_size, strides, scope=None):
        conv_block_1 = tf.reshape(i_tensor, [-1, 1, i_tensor.shape[1].value, i_tensor.shape[2].value])
        conv_block_1 = tf.nn.max_pool(conv_block_1, [1, 1, window_size, 1], [1, 1, strides, 1], 'SAME')
        conv_block_1 = tf.reshape(conv_block_1, [-1, conv_block_1.shape[2].value, conv_block_1.shape[3].value])
        return conv_block_1

    def build_inference_model(self, trainable=False):
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)

    def build_training_model(self):
        with tf.variable_scope('training', reuse=None) as scope:
            self.build_inference_model(trainable=True)
            self.output_expected = tf.placeholder(dtype='int32', shape=[None, self.max_message_length], name="OUTPUT")
            self.output_expected_oh = tf.one_hot(self.output_expected, self.byte_encoding_depth)
            self.output_expected_oh = tf.reshape(self.output_expected_oh, [-1, self.byte_encoding_depth])
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_predicted, labels=self.output_expected_oh)
            self.train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
            self.predicted_value = tf.argmax(tf.nn.softmax(self.output_predicted, 1), 1)
            self.true_value = tf.reshape(self.output_expected, [-1])
            self.batch_loss = tf.reduce_mean(loss, axis=0)
            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.true_value, tf.uint8), tf.cast(self.predicted_value, tf.uint8)), tf.float32))

    def train(self, batch_x, batch_y):
        t_1 = time.time()
        feed_dict = {self.input_layer: batch_x,
                     self.output_expected: batch_y}
        _, lo, pred, truth = tf_session().run([self.train_step, self.batch_loss, self.predicted_value, self.true_value], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        pred = np.reshape(pred, [-1, self.max_message_length])
        truth = np.reshape(truth, [-1, self.max_message_length])
        accuracy = 0
        total = 0
        for p_line, t_line in zip(pred, truth):
            char_pairs = [(x, y) for x, y in zip(p_line, t_line) if x != 0 or y != 0]
            num = sum([1 for x, y in char_pairs if x == y])
            accuracy += num
            total += len(char_pairs)

        return {'loss': float(lo), 'accuracy': float(accuracy) / total, 'time': batch_time}

    def validate(self, batch_x, batch_y):
        t_1 = time.time()
        feed_dict = {self.input_layer: batch_x, self.output_expected: batch_y}
        xxx = tf.nn.softmax(self.output_predicted, 1)
        lo, acc, pre_o, pre, truth = tf_session().run([self.batch_loss, self.batch_accuracy, xxx, self.predicted_value, self.true_value], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        pre = np.reshape(pre, [-1, self.max_message_length])
        pre_o = np.reshape(pre_o, [-1, self.max_message_length, self.byte_encoding_depth])

        for i, x in enumerate(batch_x):
            input_ = [chr(t) if 0 < t < 128 else '.' for t in x]
            output_ = [chr(t) if 0 < t < 128 else '.' for t in pre[i]]
            input_line = "".join(input_)
            output_line = "".join(output_)
            input_line = input_line.replace('\n', ' ').replace('\t', ' ').replace('\c', ' ')
            output_line = output_line.replace('\n', ' ').replace('\t', ' ').replace('\c', ' ')
            print(input_line[:100], output_line[:100])

        pred = pre
        truth = np.reshape(truth, [-1, self.max_message_length])
        accuracy = 0
        total = 0
        for p_line, t_line in zip(pred, truth):
            char_pairs = [(x, y) for x, y in zip(p_line, t_line) if x != 0 or y != 0]
            num = sum([1 for x, y in char_pairs if x == y])
            accuracy += num
            total += len(char_pairs)
        return {'loss': float(lo), 'accuracy': float(accuracy) / total, 'time': batch_time}

    def generate_line(self, probabilities):
        line = []
        for byte_probabilities in probabilities:
            probs = np.reshape(byte_probabilities, [1, 256])  # / byte_probabilities.sum()
            try:
                byte = np.random.multinomial(1, list(probs[0]), 1).argmax()
            except Exception as e:
                byte = 0
            line.append(byte)
        output_ = [chr(t) if 0 < t < 128 else '.' for t in line]
        output_ = [x if x in string.printable else '.' for x in output_]
        return "".join(output_[:128])


if __name__ == '__main__':
    foo = DataReceiver()

    def pad(array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array

    with tf_session() as session:
        trainer = Tweet2Vec_LSTM()
        trainer.build_training_model()
        session.run(tf.global_variables_initializer())
        print ('DONE')

        def train_on_batch(train_x, train_y):
            train_x = [pad(x, 256) for x in train_x]
            train_y = [pad(x, 256) for x in train_y]
            training_values = trainer.train(np.array(train_x), np.array(train_y))
            foo = "train:  Loss = {loss:.4f} --- Accuracy = {accuracy:.4f}".format(**training_values)
            print(foo)
            return training_values

        def validate_on_batch(train_x, train_y):
            train_x = [pad(x, 256) for x in train_x]
            train_y = [pad(x, 256) for x in train_y]
            training_values = trainer.validate(np.array(train_x), np.array(train_y))
            foo = "validate:  Loss = {loss:.4f} --- Accuracy = {accuracy:.4f}".format(**training_values)
            print(foo)
            return training_values

        foo.register_action_handler('train', train_on_batch)
        foo.register_action_handler('validate', validate_on_batch)
        foo.start(False)
