__name__ = "3 layer bidirectional GRU model"
__version__ = '1.0'
__author__ = 'Jean-Martin Albert'
__date__ = 'Aug 4th, 2017'
__doc__ = "A recurrent neural network for sentiment analysis of short texts"
__type__ = 'classifier'
__data__ = 'CMSUserInputDataset'
__categories__ = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Irrelevant'}

hidden_states = 128
sequence_length = 256
embedding_dimension = 256
num_classes = 4

__input__ = tf.placeholder('uint8', shape=[None, sequence_length], name="INPUT")
__output__ = None
__truth__ = None
__loss__ = None
__optimizer__ = None
__learning_rate__ = None
__optimizer_args__ = None


def multi_layer_rnn(n_layers, hidden_states):
    layers = [tf.nn.rnn_cell.GRUCell(hidden_states) for _ in range(number_of_layers)]
    if number_of_layers > 1:
        return tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)
    else:
        return layers[0]


def build_inference_model():
    global __output__
    _ = tf.one_hot(__input__, depth=embedding_dimension, axis=-1)
    _ = tf.reshape(_, [-1, sequence_length, embedding_dimension])
    fw = multi_layer_rnn(3, hidden_states)
    bw = multi_layer_rnn(3, hidden_states)
    tf.nn.bidirectional_dynamic_rnn(fw, bw, _, dtype=tf.float32)
    output, _ = tf.nn.bidirectional_dynamic_rnn(fw, bw, x, dtype=tf.float32)
    fw_output = tf.reshape(output[0][:, -1:], [-1, hidden_states])
    bk_output = tf.reshape(output[1][:, :1], [-1, hidden_states])
    v_1 = tf.contrib.layers.fully_connected(fw_output, embedding_dimension)
    v_2 = tf.contrib.layers.fully_connected(bw_output, embedding_dimension)
    e = tf.add(v_1, v_2)
    __output__ = tf.contrib.layers.fully_connected(e, num_classes)


def build_training_model(self, graph_output):
    global __truth__, __loss__
    __truth__ = tf.placeholder(dtype=tf.uint8, shape=[None, 1], name="OUTPUT")
    _ = tf.one_hot(__truth__, depth=num_classes, axis=-1)
    SXEWL = tf.nn.softmax_cross_entropy_with_logits
    __loss__ = SXEWL(logits=graph_output, labels=_)
    predicted_values = tf.cast(tf.argmax(graph_output, 1), tf.uint8)
    foo = tf.reshape(y, [-1])
    batch_loss = tf.reduce_mean(__loss__, axis=0)
    batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(foo, predicted_values), tf.float32))
    return y, loss, batch_loss, batch_accuracy


def train_setup(optimizer, learning_rate, **kwargs):
    global __optimizer__, __learning_rate__, __optimizer_args__, __loss__
    __optimizer__ = optimizer
    __learning_rate__ = learning_rate
    __optimizer_args__ = kwargs
    train_step = __optimizer__(learning_rate=__learning_rate__, **__optimizer_args__)
    return train_step.minimize(__loss__)


def pad(array, length, padding_value=0):
    array = list(array[:length])
    array += [padding_value] * (length - len(array))
    return array


def prepare_batch(batch_x, batch_y):
    batch_x = [pad(element, sequence_length) for element in batch_x]
    batch_y = [element for element in batch_y]
    return batch_x, batch_y
