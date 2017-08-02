import numpy as np
import tensorflow as tf
import sys
from models.tf_session import tf_session
from models.base_model import BaseModel, StopTraining
from stream.receiver import DataReceiver
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
            self._is_noise = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="OUTPUT")
            self._embedded_query = tf.nn.embedding_lookup(self._encode_weights, self._query_byte)
            self._embedded_context = tf.nn.embedding_lookup(self._encode_weights, self._context_byte)

        combination = tf.reduce_sum(tf.multiply(self._embedded_query, self._embedded_context), axis=1, keep_dims=True)  # tf.matmul(self.encoded_bytes, tf.transpose(self._one_hot_output))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=combination, labels=self._is_noise)
        self.accuracy = 1 - tf.reduce_mean(tf.squared_difference(tf.sigmoid(combination), self._is_noise))
        self.train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
        loss = tf.reshape(loss, [-1])
        self.batch_loss = tf.reduce_mean(loss, axis=0)

        norm = tf.sqrt(tf.reduce_sum(tf.square(self._encode_weights), 1, keep_dims=True))
        normalized_embeddings = self._encode_weights / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self._query_byte)
        self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

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
        feed_dict = {self._query_byte: batch_query, self._context_byte: batch_context, self._is_noise: batch_y}
        _, lo, acc = tf_session().run([self.train_step, self.batch_loss, self.accuracy], feed_dict=feed_dict)
        batch_time = time.time() - t_1
        return {'loss': float(lo), 'accuracy': float(acc), 'time': batch_time}

    def validate(self, batch_query, top_k=4):
        t_1 = time.time()
        feed_dict = {self._query_byte: [x[0] for x in batch_query]}
        context_bytes = [x[1] for x in batch_query]
        lo = tf_session().run([self.similarity], feed_dict=feed_dict)
        N = len(batch_query)
        i = 0
        for index, byte in enumerate(batch_query):
            nearest = (-lo[0][index, :]).argsort()[1:top_k + 1]
            s = "{num:08b} --> [{k1:08b}, {k2:08b}, {k3:08b}]"
            s = s.format(num=byte[0], k1=nearest[0], k2=nearest[1], k3=nearest[2])
            context = context_bytes[index]
            i += 1 if context in nearest else 0
        batch_time = time.time() - t_1
        return {'loss': 0, 'accuracy': 0, 'time': batch_time, 'similarity': lo[0]}

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
            foo = "Loss = {loss:.4f} --- Accuracy = {accuracy:.4f}".format(**training_values)
            return training_values

        def validate_on_batch(train_x, train_y):
            training_values = trainer.validate(np.array(train_x))
            foo = training_values
            return training_values

        foo.register_action_handler('train', train_on_batch)
        foo.register_action_handler('validate', validate_on_batch)
        foo.start(False)
