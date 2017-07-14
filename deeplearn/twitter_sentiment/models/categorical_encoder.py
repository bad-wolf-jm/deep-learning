import numpy as np
import tensorflow as tf
import sys
from models.tf_session import tf_session
from models.base_model import BaseModel, StopTraining


class CategoricalEncoder(BaseModel):
    """
    Encode a categorical variable as a vector of dimension `output_dimension`.
    """
    toplevel_scope = "categorical_encoder"

    def __init__(self, input_dimension=256, output_dimension=8):
        super(CategoricalEncoder, self).__init__()
        self._input_dtype = tf.uint8
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self._variables = {}

    def build_inference_model(self, trainable=False):
        with tf.variable_scope('input'):
            self._input = tf.placeholder(dtype=self._input_dtype, shape=[None, 1], name="INPUT")
            self._one_hot_input = tf.one_hot(self._input, depth=self.input_dimension, axis=-1)
            self._one_hot_input = tf.reshape(self._one_hot_input, [-1, self.input_dimension])

        with tf.variable_scope('encoder', reuse=True) as scope:
            weights = self.var(input_shape=[self.input_dimension, self.output_dimension],
                               name='encoding_projection', scope=scope, trainable=trainable)
            self.encoded_byte = tf.matmul(self._one_hot_input, weights)
            self._encode_weights = weights

        with tf.variable_scope('decoder', reuse=True) as scope:
            self._decoder_weights = self.var(input_shape=[self.output_dimension, self.input_dimension],
                                             name='decoding_projection', scope=scope, trainable=trainable)
            self.decoded_vector = tf.matmul(self.encoded_byte, self._decoder_weights)

    def build_training_model(self):
        self.build_inference_model(trainable=True)
        with tf.variable_scope('output'):
            self._output = tf.placeholder(dtype=self._input_dtype, shape=[None, 1], name="OUTPUT")
            self._one_hot_output = tf.one_hot(self._output, depth=self.input_dimension, axis=-1)
            self._one_hot_output = tf.reshape(self._one_hot_input, [-1, self.input_dimension])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.decoded_vector, labels=self._one_hot_output)
        loss = tf.reshape(loss, [-1, self.output_dimension])
        self.train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
        true_value = tf.argmax(self.decoded_vector, 1)
        self.seqloss = tf.reduce_mean(loss, 0)
        self.batch_loss = tf.reduce_mean(self.seqloss, axis=0)
        self.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(self._output, [-1]), tf.cast(true_value, self._input_dtype)), tf.float32))

    def encode(self, int_):
        B = np.array([int_]).reshape([1, 1])
        return self.encode_batch(B)[0]

    def encode_batch(self, batch):
        b = tf_session().run([self.encoded_byte], feed_dict={self._input: batch})
        return b[0]

    def decode(self, vector):
        return self.decode_batch([vector])

    def decode_batch(self, batch_vector):
        decoder_input = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dimension])
        decoded_vector = tf.argmax(tf.matmul(decoder_input, self._decoder_weights), 1)
        return tf_session().run([decoded_vector], feed_dict={decoder_input: batch_vector})[0]

    def get_instance_parameters(self):
        return {'input_dimension': self.input_dimension, 'output_dimension': self.output_dimension}


if __name__ == '__main__':
    if len(sys.argv) > 0:
        command = sys.argv[1]
        if command == 'train':
            with tf_session() as session:
                trainer = CategoricalEncoder(input_dimension=256, output_dimension=8)
                trainer.build_training_model()
                session.run(tf.global_variables_initializer())
                data = np.array([x for x in range(256)]).reshape([256, 1])

                max_nb_epochs = 1500
                batch_size = 16
                stop_loss = 1e-05
                num_batches_per_epoch = len(data) // batch_size
                print(num_batches_per_epoch)
                try:
                    for e in range(max_nb_epochs):
                        for batch in range(num_batches_per_epoch):
                            batch_values = data[:batch_size]
                            training_values = trainer.train(batch_values, batch_values)
                            print(training_values['accuracy'], training_values['loss'])
                            if training_values['accuracy'] == 1.0 and training_values['loss'] < stop_loss:
                                D = np.array([x for x in range(256)]).reshape([256, 1])
                                values = trainer.encode_batch(D)
                                recovered_values = trainer.decode_batch(values)
                                print(values, recovered_values)
                                keep_training = False
                                for index, point in enumerate(D):
                                    if point != recovered_values[index]:
                                        keep_training = True
                                        break
                                if not keep_training:
                                    raise StopTraining()
                            data = np.roll(data, -batch_size, axis=0)
                except StopTraining:
                    print("Encoder trained")
                trainer.save_to_pickle('weights.pkl')
        else:
            evaluator = CategoricalEncoder.instance_from_pickle('weights.pkl')
            for i in range(256):
                x = evaluator.encode(i)
                print(i, evaluator.decode(x))
