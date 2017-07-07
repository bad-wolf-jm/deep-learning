import numpy as np
import tensorflow as tf
from models.categorical_encoder import CategoricalEncoder
from models.tf_session import tf_session
from models.base_model import BaseModel  # , StopTraining
from stream.receiver import DataReceiver
import time


class Tweet2Vec_BiGRU(BaseModel):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, seq_length=1024, hidden_states=128, embedding_dimension=64, num_classes=3):
        super(Tweet2Vec_BiGRU, self).__init__()
        self._input_dtype = tf.int32
        self.seq_length = seq_length
        self.hidden_states = hidden_states
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes
        self.byte_encoder = CategoricalEncoder.instance_from_pickle('weights.pkl')

    def init_model(self, trainable=False):
        with tf.variable_scope('embedding', reuse=None) as scope:
            self._input = tf.placeholder(dtype=self._input_dtype, shape=[None, self.seq_length], name="INPUT")
            x = tf.reshape(self._input, [-1])
            x = tf.nn.embedding_lookup(self.byte_encoder._encode_weights, x)
            x = tf.reshape(x, [-1, 1024, 8])

        # NOTE self._input now has dimension [batch_size, seq_length, char_embedding_size]
        with tf.variable_scope('bidirectional_encoder', reuse=None) as scope:
            self.forward_gru = tf.nn.rnn_cell.GRUCell(self.hidden_states, activation=None)
            self.backward_gru = tf.nn.rnn_cell.GRUCell(self.hidden_states, activation=None)
            output, output_states = tf.nn.bidirectional_dynamic_rnn(self.forward_gru, self.backward_gru, x, dtype=tf.float32)
            self.fw_gru_output = tf.reshape(output[0][:, -1:], [-1, self.hidden_states])  # output[0][:,-1:]
            self.bk_gru_output = tf.reshape(output[1][:, -1:], [-1, self.hidden_states])  # output[1][:,-1:]
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