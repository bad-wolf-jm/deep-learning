import tensorflow as tf
import numpy as np


class Metadata:
    model_name = "3 layer bidirectional GRU model"
    version = '1.0'
    author = 'Jean-Martin Albert'
    date = 'Aug 4th, 2017'
    doc = "A 3 layer bidirectional recurrent neural network for sentiment analysis of short texts"
    type = 'classifier'
    data = 'CMSDataset'
    categories = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Irrelevant', 4: "OTHER"}


class Optimizer:
    name = tf.train.AdamOptimizer
    learning_rate = 0.001
    optimizer_args = {}
    minimize = None


class Hyperparameters:
    n_layers = 3
    hidden_states = 128
    sequence_length = 256
    embedding_dimension = 256
    num_classes = 5


class Globals:
    model_input = tf.placeholder('uint8', shape=[None, Hyperparameters.sequence_length], name="INPUT")
    model_output = None
    prediction = None
    truth = None
    loss = None
    batch_loss = None
    batch_accuracy = None
    minimize = None


def multi_layer_rnn(n_layers, hidden_states):
    layers = [tf.nn.rnn_cell.GRUCell(hidden_states) for _ in range(n_layers)]
    if n_layers > 1:
        return tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)
    else:
        return layers[0]


def project(input_, output_dim):
    op = tf.contrib.layers.fully_connected
    return op(input_, output_dim, biases_initializer=None)


def inference():
    _ = tf.one_hot(Globals.model_input, depth=Hyperparameters.embedding_dimension, axis=-1)
    _ = tf.reshape(_, [-1, Hyperparameters.sequence_length, Hyperparameters.embedding_dimension])
    fw = multi_layer_rnn(Hyperparameters.n_layers, Hyperparameters.hidden_states)
    bw = multi_layer_rnn(Hyperparameters.n_layers, Hyperparameters.hidden_states)
    output, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, _, dtype=tf.float32)
    fw_output = tf.reshape(output[0][:, -1:], [-1, Hyperparameters.hidden_states])
    bw_output = tf.reshape(output[1][:, :1], [-1, Hyperparameters.hidden_states])
    f = project(fw_output, Hyperparameters.embedding_dimension)
    b = project(bw_output, Hyperparameters.embedding_dimension)
    e = tf.add(f, b)
    Globals.model_output = project(e, Hyperparameters.num_classes)
    Globals.prediction = tf.cast(tf.argmax(Globals.model_output, 1), tf.uint8)
    return Globals.model_input, Globals.model_output


def loss():
    Globals.truth = tf.placeholder(dtype=tf.uint8, shape=[None, 1], name="OUTPUT")
    one_hot_encoded_truth = tf.one_hot(Globals.truth, depth=Hyperparameters.num_classes, axis=-1)
    loss_op = tf.nn.softmax_cross_entropy_with_logits
    Globals.loss = loss_op(logits=Globals.model_output, labels=one_hot_encoded_truth)
    x = tf.reshape(Globals.truth, [-1])
    Globals.batch_loss = tf.reduce_mean(Globals.loss, axis=0)
    Globals.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(Globals.prediction, x), tf.float32))
    return Globals.loss


def train():
    optimizer = Optimizer.name
    learning_rate = Optimizer.learning_rate
    args = Optimizer.optimizer_args
    Globals.minimize = optimizer(learning_rate, **args).minimize(loss())


def pad(array, length, padding_value=0):
    array = list(array[:length])
    array += [padding_value] * (length - len(array))
    return array


def prepare_batch(batch_x, batch_y):
    batch_x = [pad(element, Hyperparameters.sequence_length) for element in batch_x]
    batch_y = [element for element in batch_y]
    return {Globals.model_input: batch_x,
            Globals.truth: batch_y}


def evaluate_batch(batch_x, batch_y, session=None):
    feed_dict = prepare_batch(batch_x, batch_y)
    p, loss, accuracy = session.run([Globals.prediction, Globals.batch_loss, Globals.batch_accuracy],
                                    feed_dict=feed_dict)
    _ = [x[0] for x in batch_y]
    d = {'loss': loss, 'accuracy': accuracy, 'output': zip(batch_x, _, p)}
    return d


def train_batch(batch_x, batch_y, session=None):
    return session.run(Globals.minimize, feed_dict = prepare_batch(batch_x, batch_y))


def compute(batch_x, session=None):
    feed_dict = prepare_batch(batch_x, [])
    p = session.run([Globals.prediction], feed_dict=feed_dict)
    return p[0]
