import numpy as np
import tensorflow as tf
from models.categorical_encoder import CategoricalEncoder
from models.tf_session import tf_session
from models.base_model import BaseModel  # , StopTraining
from train.stream import DataReceiver
import time

EMBEDDING_DIMENSION = 6
NUM_FILTERS = 3


class Tweet2Vec_LSTM(BaseModel):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, seq_length=1024, hidden_states=128, embedding_dimension=64, num_classes=3):
        super(Tweet2Vec_LSTM, self).__init__()
        self._input_dtype = tf.int32
        self.seq_length = seq_length
        self.hidden_states = hidden_states
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes
        self.byte_encoder = CategoricalEncoder.instance_from_pickle('weights.pkl')

    def init_model(self, trainable=False):
        pkeep = 0.7
        NLAYERS = 2
        LSTM_SIZE = 128
        DECODER_SIZE = 128
        INTERNALSIZE = 128
        Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE * NLAYERS], name='Hin')
        with tf.variable_scope('input', reuse=None) as scope:
            input_layer = tf.placeholder('uint8', shape=[None, self.input_width], name="INPUT")

        with tf.variable_scope('encoder', reuse=None) as scope:
            with tf.variable_scope('convolution', reuse=None) as scope:
                conv_1 = self.convolutional_block(self._input, EMBEDDING_DIMENSION, NUM_FILTERS)  # tf.nn.conv1d(input_layer, num_filters, 7, convolution=convops)
                pool_1 = self.max_pool_layer(conv_1, 3, 1)
                conv_2 = self.convolutional_block(pool_1, NUM_FILTERS, 7)  # tf.nn.conv1d(input_layer, num_filters, 7, convolution=convops)
                pool_2 = self.max_pool_layer(conv_2, 3, 1)
                conv_3 = self.convolutional_block(pool_2, NUM_FILTERS, 3)
                conv_4 = self.convolutional_block(conv_3, NUM_FILTERS, 3)  # tf.nn.conv1d(conv_3, NUM_FILTERS, 3)

            with tf.variable_scope('lstm_encoder', reuse=None) as scope:
                encoding_layer = tf.nn.rnn.BasicLSTMCell(LSTM_SIZE)  # LSTM(conv_4, encoder_size, no_return_seq=True, name='encoder')
                Yr, H = tf.nn.dynamic_rnn(encoding_layer, conv_4, dtype=tf.float32, initial_state=Hin)  # l_in_rep = Repeat(encoding_layer, n=max_chars)

        with tf.variable_scope('decoder', reuse=None) as scope:
            cells = [tf.nn.rnn.LSTMCell(DECODER_SIZE) for _ in range(NLAYERS)]
            dropcells = [tf.nn.rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
            multicell = tf.nn.rnn.MultiRNNCell(dropcells, state_is_tuple=False)
            multicell = tf.nn.rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
            Yr, H = tf.nn.dynamic_rnn(multicell, self._encoding_layer, dtype=tf.float32, initial_state=Hin)

    def convolutional_block(self, i_tensor, i_features, o_features, scope=None):
        with tf.variable_scope(scope):
            kernel = self.var(input_shape=[3, i_features, o_features], name='layer_1_convolution')
            x = tf.nn.conv1d(i_tensor, kernel, stride=1, padding="SAME")
            conv = tf.nn.relu(x)
            return conv

    def max_pool_layer(self, i_tensor, window_size, strides):
        conv_block_1 = tf.reshape(i_tensor, [-1, 1, i_tensor.shape[1].value, i_tensor.shape[2].value])
        conv_block_1 = tf.nn.max_pool(conv_block_1, [1, 1, window_size, 1], [1, 1, strides, 1], 'SAME')
        conv_block_1 = tf.reshape(conv_block_1, [-1, conv_block_1.shape[2].value, conv_block_1.shape[3].value])
        return conv_block_1

    def build_inference_model(self, trainable=False):
        with tf.variable_scope('inference', reuse=None):
            self.init_model(trainable=trainable)

    def build_training_model(self):
        y_flat = T.flatten(y, 1)
        y_flat_idx = T.arange(y_flat.shape[0]) * char_vocab_size + y_flat
        costf = -T.log(prediction.flatten()[y_flat_idx] + 1e-8)
        costf = costf.reshape([y.shape[0], y.shape[1]])
        costf = (costf * y_mask).sum(0)
        loss = costf.sum()

        with tf.variable_scope('training', reuse=None) as scope:
            self.build_inference_model(trainable=True)
            self.final_layer_weights = self.var(input_shape=[self.embedding_dimension, self.num_classes],
                                                name='final_weights',
                                                scope=scope,
                                                trainable=True)
            self.output_predicted = tf.matmul(self.enc_output, self.final_layer_weights)
            self.output_expected = tf.placeholder(dtype=self._input_dtype, shape=[None, 1], name="INPUT")
            self.output_expected_oh = tf.one_hot(self.output_expected, depth=self.num_classes, axis=-1)  # tf.nn.embedding_lookup(self.byte_encoder._encode_weights, self.output_expected)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_predicted, labels=self.output_expected_oh)
            self.train_step = tf.train.MomentumOptimizer(learning_rate=0.000001, momentum=0.9).minimize(loss)
            self.predicted_value = tf.argmax(self.output_predicted, 1)
            self.true_value = tf.reshape(self.output_expected, [-1])
            self.batch_loss = tf.reduce_mean(loss, axis=0)
            self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.true_value, tf.uint8), tf.cast(self.predicted_value, tf.uint8)), tf.float32))

    def train(self, batch_x, batch_y):
        t_1 = time.time()
        feed_dict = {self._input: batch_x, self.output_expected: batch_y}
        _, lo, acc = tf_session().run([self.train_step, self.batch_loss, self.batch_accuracy], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': float(lo), 'accuracy': float(acc), 'time': batch_time}


if __name__ == '__main__':
    foo = DataReceiver()

    def pad(array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array

    with tf_session() as session:
        trainer = Tweet2Vec_BiGRU()
        trainer.build_training_model()
        session.run(tf.global_variables_initializer())
        print ('DONE')

        def train_on_batch(train_x, train_y):
            train_x = [pad(x, 1024) for x in train_x]
            training_values = trainer.train(np.array(train_x), np.array(train_y))
            foo = "train:  Loss = {loss:.4f} --- Accuracy = {accuracy:.4f}".format(**training_values)
            print(foo)
            return training_values

        def validate_on_batch(train_x, train_y):
            training_values = trainer.validate(np.array(train_x))
            foo = "validate:  Loss = {loss:.4f} --- Accuracy = {accuracy:.4f}".format(**training_values)
            print(foo)
            return training_values

        foo.register_action_handler('train', train_on_batch)
        foo.start(False)
