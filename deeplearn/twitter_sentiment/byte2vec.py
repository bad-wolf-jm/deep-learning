import numpy as np
import tensorflow as tf
import sys
from tf_session import tf_session
from base_model import BaseModel, StopTraining
from train.stream import DataReceiver
import time

class Byte2Vec(BaseModel):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, input_dimension=256, output_dimension=8):
        super(Byte2Vec, self).__init__()
        self._input_dtype = tf.int32
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self._variables = {}

    def build_inference_model(self, trainable=False):
        with tf.variable_scope('embedding', reuse=True) as scope:
            self._encode_weights = self.var(input_shape=[self.input_dimension, self.output_dimension],
                                            name='embedding_matrix', scope=scope, trainable=trainable)

    def build_training_model(self):
        self.build_inference_model(trainable=True)
        with tf.variable_scope('output'):
            self._query_byte = tf.placeholder(dtype=self._input_dtype, shape=[None], name="QUERY")
            self._context_byte = tf.placeholder(dtype=self._input_dtype, shape=[None], name="CONTEXT")
            self._is_noise = tf.placeholder(dtype=tf.float32, shape=[None,1], name="OUTPUT")
            self._embedded_query = tf.nn.embedding_lookup(self._encode_weights, self._query_byte)
            self._embedded_context = tf.nn.embedding_lookup(self._encode_weights, self._context_byte)
            self._embedded_context = tf.reshape(self._embedded_context, [-1, self.output_dimension])

        combination = tf.reduce_sum( tf.multiply(self._embedded_query, self._embedded_context), axis = 1, keep_dims=True ) #tf.matmul(self.encoded_bytes, tf.transpose(self._one_hot_output))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=combination, labels=self._is_noise)
        self.train_step = tf.train.RMSPropOptimizer(0.0000001).minimize(loss)
        loss = tf.reshape(loss, [-1])
        self.batch_loss = tf.reduce_mean(loss, axis=0)

    def encode(self, int_):
        B = np.array([int_]).reshape([1, 1])
        return self.encode_batch(B)[0]

    def encode_batch(self, batch):
        lookup = tf.nn.embedding_lookup(self._encode_weights, batch)
        b = tf_session().run([loopup], feed_dict={self._input: batch})
        return b[0]

    def decode(self, vector):
        return self.decode_batch([vector])

    def decode_batch(self, batch_vector):
        decoder_input = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dimension])
        decoded_vector = tf.argmax(tf.matmul(decoder_input, self._decoder_weights), 1)
        return tf_session().run([decoded_vector], feed_dict={decoder_input: batch_vector})[0]

    def get_instance_parameters(self):
        return {'input_dimension': self.input_dimension, 'output_dimension': self.output_dimension}

    def train(self, batch_query, batch_context, batch_y):
        t_1 = time.time()
        feed_dict = {self._query_byte: batch_query, self._context_byte:batch_context, self._is_noise: batch_y}
        _, lo = tf_session().run([self.train_step, self.batch_loss], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': lo, 'time': batch_time}

if __name__ == '__main__':
    foo = DataReceiver()
    with tf_session() as session:
        trainer = Byte2Vec(input_dimension=256, output_dimension=16)
        trainer.build_training_model()
        session.run(tf.global_variables_initializer())

        def train_on_batch(train_x, train_y):
            query = [x[0] for x in train_x]
            context = [x[1] for x in train_x]
            training_values = trainer.train(np.array(query), np.array(context), np.array(train_y))
            print(training_values)

        foo.register_action_handler('train', train_on_batch)
        foo.start(False)
