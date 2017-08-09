import tensorflow as tf
import numpy as np
import sys

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
    n_layers = 2
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
    #print(state)
    Globals.encoder_output = state
    with tf.variable_scope('decoder'):
        training_decoder_input = tf.placeholder('uint8', shape=[None, Hyperparameters.sequence_length], name="INPUT")
        _ = tf.one_hot(training_decoder_input, depth=Hyperparameters.embedding_dimension, axis=-1)
        _ = tf.reshape(_, [-1, Hyperparameters.sequence_length, Hyperparameters.embedding_dimension])
        decode = multi_layer_rnn(Hyperparameters.n_layers, Hyperparameters.hidden_states)
        decoded_output, state = tf.nn.dynamic_rnn(decode, _, dtype=tf.float32, initial_state=state)
        #print(decoded_output.get_shape())
        decoded_output = tf.reshape(decoded_output, [-1, Hyperparameters.hidden_states])
        output = project(decoded_output, Hyperparameters.embedding_dimension)
        #print(output.get_shape())
        out = tf.cast(tf.argmax(output, 1), tf.uint8)
        out = tf.reshape(out, [-1, Hyperparameters.sequence_length])
        Globals.training_decoder_input = training_decoder_input
        Globals.model_output = output
        Globals.prediction = out
        Globals.decoder = decode
        Globals.decoder_input = _
        #Globals.decoder_state = decoded_output

def decoder():
    with tf.variable_scope('compute-decoder'):
        c_input = tf.placeholder('uint8', shape=[None, 1], name="INPUTSSS")
        _ = tf.one_hot(c_input, depth=Hyperparameters.embedding_dimension, axis=-1)
        i = tf.reshape(_, [-1, 1, Hyperparameters.embedding_dimension])
        decoder_initial_state = tf.placeholder('float32', shape=[Hyperparameters.n_layers, None, Hyperparameters.hidden_states], name="INPUT")
        #transposed_initial_state = tf.transpose(decoder_initial_state, perm=[1,0,2])
        states = tuple([decoder_initial_state[i] for i in range(Hyperparameters.n_layers)])
        decoded_output, state = tf.nn.dynamic_rnn(Globals.decoder, i, dtype=tf.float32, initial_state=states)
        decoded_output = tf.reshape(decoded_output, [-1, Hyperparameters.hidden_states])
        #print(decoded_output)
        output = project(decoded_output, Hyperparameters.embedding_dimension)
        print(output)
        out = tf.cast(tf.argmax(output, 1), tf.uint8)
        out = tf.reshape(out, [-1, 1])
        print(out)
        Globals.compute_input = c_input
        Globals.decoder_initial_state = decoder_initial_state
        Globals.compute_output = out
        Globals.compute_state = decoder_initial_state
        #Globals.decoder_input = decoder_one_hot_input


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
    #print(d)
    return d


def train_batch(batch_x, batch_y, session=None):
    return session.run(Globals.minimize, feed_dict = prepare_batch(batch_x, batch_y))


def compute(batch_x, session=None):
    feed_dict = prepare_batch(batch_x, [])
    encoded = session.run(Globals.encoder_output, feed_dict={Globals.model_input: batch_x})
    #print('XXX=', encoded[0].shape)
    i_state = np.array(encoded)
    #i_state = np.transpose()
    input_ = [[0]]*len(batch_x)
    #feed_dict[Globals.decoder_initial_state] = i_state
    columns = []
    for i in range(Hyperparameters.sequence_length):
        output, new_state = session.run([Globals.compute_output, Globals.compute_state],
                                        feed_dict={Globals.compute_input: input_,
                                                   Globals.compute_state: i_state})
        #print('output=', output)
        i_state = new_state
        input_ = output
        columns.append(output)
    return np.transpose(np.array(columns))
    #p = session.run([Globals.prediction], feed_dict=feed_dict)
    #return p[0]


if __name__ == '__main__':
    from prettytable import PrettyTable
    with tf.Session() as session:
        inference()
        train()
        decoder()
        session.run(tf.global_variables_initializer())
        for i in range(100000):
            batch = np.random.randint(1, Hyperparameters.num_classes, size=[10, Hyperparameters.sequence_length])
            batch_x = [x for x in batch]
            batch_y = [sorted(x) for x in batch]
            train_batch(batch_x, batch_y, session=session)
            sys.stderr.write('x')
            sys.stderr.flush()
            #print (foo[0].shape)
            if i % 100 == 0:
                d = evaluate_batch(batch_x, batch_y, session=session)
                #foo = compute(batch_x, session=session)
                x = PrettyTable(["Original", "Sorted"])
                x.align["Original"] = "l"
                x.align["Sorted"] = "l"
                for orig, sor, q in zip(batch_x,  batch_y, [x[2] for x in d['output']]):
                    try:
                        str_1 = "[{}]".format(" ".join([str(x) for x in orig]))
                        str_2 = "[{}]".format(" ".join([str(x) if orig[i]== x else '__' for i, x in enumerate(q)]))
                        x.add_row([str_1, str_2])
                    except:
                        pass
                print('Loss:', d['loss'], ' --- ', 'Accuracy:', d['accuracy'])
                print (x)
        #pass
