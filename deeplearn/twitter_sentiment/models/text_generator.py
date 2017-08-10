import tensorflow as tf
import os
import numpy as np


class Metadata:
    model_name = "GRU based text generator"
    version = '1.0'
    author = 'Jean-Martin Albert'
    date = 'Aug 4th, 2017'
    doc = "A text generator"
    type = 'generator'
    data = 'HarryPotterDataset'


class Optimizer:
    name = tf.train.AdamOptimizer
    learning_rate = 0.0001
    optimizer_args = {}
    minimize = None


class Hyperparameters:
    n_layers = 3
    hidden_states = 512
    sequence_length = 30
    embedding_dimension = 128
    dropout_keep = 0.8


class Globals:
    model_input = tf.placeholder('uint8', shape=[None, Hyperparameters.sequence_length], name="INPUT")
    initial_state = tf.placeholder(tf.float32, [None, Hyperparameters.n_layers, Hyperparameters.hidden_states], name='Hin')
    model_output = None
    prediction = None
    truth = None
    loss = None
    batch_loss = None
    batch_accuracy = None
    minimize = None


def multi_layer_rnn(n_layers, hidden_states):
    _ = [tf.nn.rnn_cell.GRUCell(hidden_states) for _ in range(n_layers)]
    _ = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in _]
    if n_layers > 1:
        _ = tf.nn.rnn_cell.MultiRNNCell(_, state_is_tuple=True)
        _ = rnn.DropoutWrapper(_, output_keep_prob=pkeep)
        return _
    else:
        return _[0]


def project(input_, output_dim):
    op = tf.contrib.layers.fully_connected
    return op(input_, output_dim, biases_initializer=None)


def inference():
    _ = tf.one_hot(Globals.model_input, depth=Hyperparameters.embedding_dimension, axis=-1)
    _ = tf.reshape(_, [-1, Hyperparameters.sequence_length, Hyperparameters.embedding_dimension])
    encode = multi_layer_rnn(Hyperparameters.n_layers, Hyperparameters.hidden_states)
    output, state = tf.nn.dynamic_rnn(encode, _, dtype=tf.float32, initial_state=Globals.initial_state)
    Globals.model_output = output
    output = tf.reshape(output, [-1, Hyperparameters.hidden_states])
    output = project(decoded_output, Hyperparameters.embedding_dimension)
    out = tf.cast(tf.argmax(output, 1), tf.uint8)
    out = tf.reshape(out, [-1, Hyperparameters.sequence_length])
    Globals.generated_sequence = out
    Globals.model_output = output


def loss():
    Globals.truth = tf.placeholder(dtype=tf.uint8, shape=[None, Hyperparameters.sequence_length], name="OUTPUT")
    one_hot_encoded_truth = tf.one_hot(Globals.truth, depth=Hyperparameters.num_classes, axis=-1)
    loss_op = tf.nn.softmax_cross_entropy_with_logits
    Globals.loss = loss_op(logits=Globals.model_output, labels=one_hot_encoded_truth)
    x = tf.reshape(Globals.truth, [-1, Hyperparameters.sequence_length])
    Globals.batch_loss = tf.reduce_mean(Globals.loss, axis=0)
    Globals.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(Globals.prediction, x), tf.float32))
    return Globals.loss


def train():
    optimizer = Optimizer.name
    learning_rate = Optimizer.learning_rate
    args = Optimizer.optimizer_args
    Globals.minimize = optimizer(learning_rate, **args).minimize(loss())


if __name__ == '__main__':

    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state
    with tf.Session() as session:
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        step = 0

        # training loop
        for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=10):
            # train on one minibatch
            feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
            _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

            # loop state around
            istate = ostate
            step += BATCHSIZE * SEQLEN
