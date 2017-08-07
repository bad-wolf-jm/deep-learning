import tensorflow as tf
import numpy as np

__name__ = "3 layer bidirectional GRU model"
__version__ = '1.0'
__author__ = 'Jean-Martin Albert'
__date__ = 'Aug 4th, 2017'
__doc__ = "A 3 layer bidirectional recurrent neural network for sentiment analysis of short texts"
__type__ = 'classifier'
__data__ = 'CMSDataset'
__categories__ = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Irrelevant', 4: 'Irrelevant'}

hidden_states = 128
sequence_length = 256
embedding_dimension = 256
num_classes = 5

__input__ = tf.placeholder('uint8', shape=[None, sequence_length], name="INPUT")
__output__ = None
__truth__ = None
__loss__ = None

__optimizer__ = 'adam'
__learning_rate__ = 0.001
__optimizer_args__ = {}


def multi_layer_rnn(n_layers, hidden_states):
    layers = [tf.nn.rnn_cell.GRUCell(hidden_states) for _ in range(n_layers)]
    if n_layers > 1:
        return tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)
    else:
        return layers[0]


def project(x, n_dims):
    return tf.contrib.layers.fully_connected(x, n_dims, biases_initializer=None)


def inference():
    global __input__, __output__, __prediction__
    _ = tf.one_hot(__input__, depth=embedding_dimension, axis=-1)
    _ = tf.reshape(_, [-1, sequence_length, embedding_dimension])
    fw = multi_layer_rnn(3, hidden_states)
    bw = multi_layer_rnn(3, hidden_states)
    output, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, _, dtype=tf.float32)
    fw_output = tf.reshape(output[0][:, -1:], [-1, hidden_states])
    bw_output = tf.reshape(output[1][:, :1], [-1, hidden_states])
    v_1 = project(fw_output, embedding_dimension)
    v_2 = project(bw_output, embedding_dimension)
    e = tf.add(v_1, v_2)
    __output__ = project(e, num_classes)
    __prediction__ = tf.cast(tf.argmax(__output__, 1), tf.uint8)
    return __input__, __output__


def loss():
    global __truth__, __loss__, __batch_loss__, __batch_accuracy__, __prediction__
    __truth__ = tf.placeholder(dtype=tf.uint8, shape=[None, 1], name="OUTPUT")
    _ = tf.one_hot(__truth__, depth=num_classes, axis=-1)
    SXEWL = tf.nn.softmax_cross_entropy_with_logits
    __loss__ = SXEWL(logits=__output__, labels=_)
    _ = tf.reshape(__truth__, [-1])
    __batch_loss__ = tf.reduce_mean(__loss__, axis=0)
    __batch_accuracy__ = tf.reduce_mean(tf.cast(tf.equal(__prediction__, _), tf.float32))
    return __loss__


def pad(array, length, padding_value=0):
    array = list(array[:length])
    array += [padding_value] * (length - len(array))
    return array


def prepare_batch(batch_x, batch_y):
    batch_x = [pad(element, sequence_length) for element in batch_x]
    batch_y = [element for element in batch_y]
    return {__input__: batch_x,
            __truth__: batch_y}


def test(train_x, train_y, session=None):
    feed_dict = prepare_batch(train_x, train_y)
    p_v, lo, acc = session.run([__prediction__, __batch_loss__, __batch_accuracy__], feed_dict=feed_dict)
    _ = [x[0] for x in train_y]
    d = {'loss': lo, 'accuracy': acc, 'output': zip(train_x, _, p_v)}
    return d


def train(train_x, train_y, session=None):
    feed_dict = prepare_batch(train_x, train_y)
    _, b_loss, b_accuracy = session.run([__train_step__, __batch_loss__, __batch_accuracy__], feed_dict=feed_dict)
    d = {'loss': b_loss, 'accuracy': b_accuracy, 'output': []}
    return d


def validate(train_x, train_y, session=None):
    feed_dict = prepare_batch(train_x, train_y)
    loss, accuracy = session.run([__batch_loss__, __batch_accuracy__], feed_dict=feed_dict)
    d = {'loss': loss, 'accuracy': accuracy, 'output': []}
    return d
