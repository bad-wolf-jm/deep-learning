import tensorflow as tf
import os
import numpy as np
import glob
import string


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
    model_input = tf.placeholder('uint8', shape=[None, None], name="INPUT")
    initial_state = tf.placeholder(tf.float32, [Hyperparameters.n_layers, None,  Hyperparameters.hidden_states], name='Hin')
    model_output = None
    prediction = None
    truth = None
    loss = None
    batch_loss = None
    batch_accuracy = None
    minimize = None


def multi_layer_rnn(n_layers, hidden_states):
    _ = [tf.nn.rnn_cell.GRUCell(hidden_states) for _ in range(n_layers)]
    _ = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=Hyperparameters.dropout_keep) for cell in _]
    if n_layers > 1:
        _ = tf.nn.rnn_cell.MultiRNNCell(_, state_is_tuple=True)
        _ = tf.nn.rnn_cell.DropoutWrapper(_, output_keep_prob=Hyperparameters.dropout_keep)
        return _
    else:
        return _[0]


def project(input_, output_dim):
    op = tf.contrib.layers.fully_connected
    return op(input_, output_dim, biases_initializer=None)


def inference():
    _ = tf.one_hot(Globals.model_input, depth=Hyperparameters.embedding_dimension, axis=-1)
    encode = multi_layer_rnn(Hyperparameters.n_layers, Hyperparameters.hidden_states)
    state_tuple = tuple(tf.unstack(Globals.initial_state, axis=0))
    output, state = tf.nn.dynamic_rnn(encode, _, dtype=tf.float32, initial_state=state_tuple)
    output = tf.reshape(output, [-1, Hyperparameters.hidden_states])
    output = project(output, Hyperparameters.embedding_dimension)
    out = tf.cast(tf.argmax(output, 1), tf.uint8)
    out = tf.reshape(out, [-1, Hyperparameters.sequence_length])
    Globals.generated_sequence = out
    Globals.generated_characters = tf.nn.softmax(output)
    Globals.model_output = output
    Globals.state = state


def loss():
    Globals.truth = tf.placeholder(dtype=tf.uint8, shape=[None, Hyperparameters.sequence_length], name="OUTPUT")
    one_hot_encoded_truth = tf.one_hot(Globals.truth, depth=Hyperparameters.embedding_dimension, axis=-1)
    loss_op = tf.nn.softmax_cross_entropy_with_logits
    Globals.loss = loss_op(logits=Globals.model_output, labels=one_hot_encoded_truth)
    x = tf.reshape(Globals.truth, [-1, Hyperparameters.sequence_length])
    Globals.batch_loss = tf.reduce_mean(Globals.loss, axis=0)
    Globals.batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(Globals.generated_sequence, x), tf.float32))
    return Globals.loss


def train():
    optimizer = Optimizer.name
    learning_rate = Optimizer.learning_rate
    args = Optimizer.optimizer_args
    Globals.minimize = optimizer(learning_rate, **args).minimize(loss())


# Temporary place for data reader


def read_file(file_name):
    file_ = open(file_name, "r")
    end_of_sentence = True
    line = file_.readline()
    content = []
    num_chars = 0
    while line != "":
        line = line.replace('\t', ' ')
        if line == '\n':
            if end_of_sentence:
                content.extend('\n\n')
                num_chars += 1
            while line == '\n':
                line = file_.readline()
        try:
            end_of_sentence = (line[-2] in '.!?')
        except:
            end_of_sentence = False
        line = line[:-1] + ' '
        content.extend([x for x in line if ord(x) < 128])
        num_chars += len(line)
        line = file_.readline()
    file_.close()
    return content


def read_data_files_from_folder(directory, validation=0.1):
    codetext = []
    bookranges = []
    file_start = 0
    print(directory)
    for file_name in glob.glob(directory, recursive=True):
        file_ = open(file_name, "r")
        print("Loading file: " + file_name)
        chars = read_file(file_name)
        codetext.extend([ord(x) for x in chars if ord(x) < 128])
        num_chars = len(chars)
        bookranges.append({"start": file_start,
                           "end": file_start + num_chars,
                           "name": file_name.rsplit("/", 1)[-1]})
        file_start += num_chars
    return codetext


def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array([x for x in raw_data])
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)

    total_num_batches = (data_len * nb_epochs) // batch_size
    #batches_per_epoch = len(data_x) // batch_size

    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])
    I = 0
    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            I += 1
            yield {'train_x':  x,
                   'train_y':  y,
                   'validate': None,
                   'batch_number':  batch,
                   'epoch_number':  epoch+1,
                   'batch_index':   I,
                   'total_batches': total_num_batches,
                   'total_epochs':  nb_epochs}


def generate_text(length, session=None):
    generated_text = ''
    character = [[ord(' ')]]
    print('Generating text')
    istate = np.zeros([Hyperparameters.n_layers, 1, Hyperparameters.hidden_states])
    while len(generated_text) < length:
        feed_dict = {Globals.model_input: character, Globals.initial_state: istate}
        next_char, state = session.run([Globals.generated_characters, Globals.state], feed_dict=feed_dict)
        next_char = np.asarray(next_char).astype('float64')
        next_char = next_char / next_char.sum()
        next_char_id = np.random.multinomial(1, next_char.squeeze(), 1).argmax()
        next_char_id = next_char_id if chr(next_char_id) in string.printable else ord(" ")
        generated_text += chr(next_char_id)
        character = [[next_char_id]]
        istate = state
    return generated_text


if __name__ == '__main__':
    root = os.path.dirname(__file__)
    root = os.path.join(root, 'text_gen_data')
    save = os.path.join(os.path.expanduser('~'), '.text_gen', 'output')
    chkpt = os.path.join(os.path.expanduser('~'), '.text_gen', 'model_chkpt')
    if not os.path.exists(save):
        os.makedirs(save)
    data = read_data_files_from_folder(os.path.join(root, 'harry_potter/*.txt'))
    inference()
    train()
    index = 0
    istate = np.zeros([Hyperparameters.n_layers, 32, Hyperparameters.hidden_states])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if not os.path.exists(chkpt):
            os.makedirs(chkpt)
        else:
            s = tf.train.Saver()
            s.save(session, os.path.join(chkpt, "t_gen_weights"))
            print('Model restored')

        for batch in rnn_minibatch_sequencer(data, 32, Hyperparameters.sequence_length, nb_epochs=1000):
            x = batch['train_x']
            y = batch['train_y']
            feed_dict = {Globals.model_input: x,
                         Globals.truth: y,
                         Globals.initial_state: istate}
            _, ostate = session.run([Globals.minimize, Globals.state], feed_dict=feed_dict)
            print('x', end='', flush=True)
            istate = ostate
            #print(index, batch['epoch_number'])
            if index % (batch['epoch_number']*150) == 0:
                file_name = "text_gen-{}".format(index)
                file_name = os.path.join(save, file_name)
                with open(file_name, 'w') as test_file:
                    test_file.write(generate_text(4000, session))
            if index % 250 == 0:
                s = tf.train.Saver()
                s.save(session, os.path.join(chkpt, "t_gen_weights"))
            index += 1
