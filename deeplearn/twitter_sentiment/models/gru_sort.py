import tensorflow as tf
import numpy as np
import sys
import os


class Metadata:
    model_name = "Sorting function based on recurrent neural networks"
    version = '1.0'
    author = 'Jean-Martin Albert'
    date = 'Aug 8th, 2017'
    doc = "This network sorts short sequences of integers"
    type = 'encoder'
    data = 'ShortSequenceSorter'


class Optimizer:
    name = tf.train.AdamOptimizer
    learning_rate = 0.0001
    optimizer_args = {}
    minimize = None


class Hyperparameters:
    n_layers = 4
    hidden_states = 512
    sequence_length = 32
    embedding_dimension = 128
    num_classes = 128


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
    encode = multi_layer_rnn(Hyperparameters.n_layers, Hyperparameters.hidden_states)
    encoded_input, state = tf.nn.dynamic_rnn(encode, _, dtype=tf.float32)
    Globals.encoder_output = state
    with tf.variable_scope('decoder'):
        training_decoder_input = tf.zeros_like(Globals.model_input) #placeholder('uint8', shape=[None, Hyperparameters.sequence_length], name="INPUT")
        _ = tf.one_hot(training_decoder_input, depth=Hyperparameters.embedding_dimension, axis=-1)
        _ = tf.reshape(_, [-1, Hyperparameters.sequence_length, Hyperparameters.embedding_dimension])
        decode = multi_layer_rnn(Hyperparameters.n_layers, Hyperparameters.hidden_states)
        decoded_output, state = tf.nn.dynamic_rnn(decode, _, dtype=tf.float32, initial_state=state)
        decoded_output = tf.reshape(decoded_output, [-1, Hyperparameters.hidden_states])
        output = project(decoded_output, Hyperparameters.embedding_dimension)
        out = tf.cast(tf.argmax(output, 1), tf.uint8)
        out = tf.reshape(out, [-1, Hyperparameters.sequence_length])
        Globals.training_decoder_input = training_decoder_input
        Globals.model_output = output
        Globals.prediction = out
        Globals.decoder = decode
        Globals.decoder_input = _


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


def pad(array, length, padding_value=0):
    array = list(array[:length])
    array += [padding_value] * (length - len(array))
    return array


def prepare_batch(batch_x, batch_y):
    batch_x = [pad(element, Hyperparameters.sequence_length) for element in batch_x]
    batch_y_t = [list([0 for _ in range(Hyperparameters.sequence_length)]) for element in batch_y]
    batch_y = [pad(element, Hyperparameters.sequence_length) for element in batch_y]
    return {Globals.model_input: batch_x,
            Globals.training_decoder_input: batch_y_t,
            Globals.truth: batch_y}


def evaluate_batch(batch_x, batch_y, session=None):
    feed_dict = prepare_batch(batch_x, batch_y)
    p, loss, accuracy = session.run([Globals.prediction, Globals.batch_loss, Globals.batch_accuracy], feed_dict=feed_dict)
    _ = [x[0] for x in batch_y]
    d = {'loss': loss, 'accuracy': accuracy, 'output': zip(batch_x, _, p)}
    return d


def train_batch(batch_x, batch_y, session=None):
    return session.run(Globals.minimize, feed_dict=prepare_batch(batch_x, batch_y))


def compute(batch_x, session=None):
    params = [pad(element, Hyperparameters.sequence_length) for element in batch_x]
    p = session.run(Globals.prediction, feed_dict={Globals.model_input: params})
    return p


if __name__ == '__main__':
    save = os.path.join(os.path.expanduser('~'), '.gru_sort')
    if not os.path.exists(save):
        os.makedirs(save)
    from prettytable import PrettyTable
    with tf.Session() as session:
        inference()
        train()
        session.run(tf.global_variables_initializer())
        index = 1
        for i in range(500000):
            batch = np.random.randint(1, Hyperparameters.num_classes, size=[10, Hyperparameters.sequence_length])
            batch_x = [x for x in batch]
            batch_y = [sorted(x) for x in batch]
            train_batch(batch_x, batch_y, session=session)
            sys.stderr.write('x')
            sys.stderr.flush()
            if i % 250 == 0:
                batch = np.random.randint(1, Hyperparameters.num_classes, size=[100, Hyperparameters.sequence_length])
                batch_x = [x for x in batch]
                batch_y = [sorted(x) for x in batch]
                d = evaluate_batch(batch_x, batch_y, session=session)
                x = PrettyTable(["Original", "Sorted"])
                x.align["Original"] = "l"
                x.align["Sorted"] = "l"
                file_name = "sort-test-{}".format(index)
                file_name = os.path.join(save, file_name)
                with open(file_name, 'w') as test_file:
                    for orig, sor, q in zip(batch_x,  batch_y, [x[2] for x in d['output']]):
                        try:
                            str_1 = "{}".format(",".join([str(x) for x in orig]))
                            str_2 = "{}".format(",".join([str(x) for x in q]))
                            line = '{str_1};{str_2}\n'.format(str_1=str_1, str_2=str_2)
                            test_file.write(line)
                        except:
                            pass

                        try:
                            str_1 = "[{}]".format(" ".join([str(x) for x in orig]))
                            str_2 = "[{}]".format(" ".join([str(x) if sor[i] == x else '___' for i, x in enumerate(q)]))
                            x.add_row([str_1, str_2])
                        except:
                            pass
                index += 1
                print('Loss:', d['loss'], ' --- ', 'Accuracy:', d['accuracy'])
                print (x)
        # pass
