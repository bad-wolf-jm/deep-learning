import numpy as np


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
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch
