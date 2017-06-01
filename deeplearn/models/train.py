# training utilities for keras networks
import os
import sys
import numpy as np
import logging
import time


def train_test_split_indices(total_number, test_fraction):
    """
    This function produces a random list of indices drawn uniformly from the list 'range(total_number)'.
    It returns about 'test_fraction*total_number' indices, which are suitable for splitting a set into
    a training and a test set.
    """
    return set(list(np.random.choice(total_number, size = [int(test_fraction * total_number)], replace = False)))

def basic_train_test_split(data_x, data_y, fraction):
    test_indices = train_test_split_indices(len(data_x), fraction)
    train_in  = []
    train_out = []
    test_in   = []
    test_out  = []
    for index, point in enumerate(data_x):
        if index in test_indices:
            test_in.append(data_x[index])
            test_out.append(data_y[index])
        else:
            train_in.append(data_x[index])
            train_out.append(data_y[index])
    return {'train': {'input': np.array(train_in), 'output': np.array(train_out)},
            'test':  {'input': np.array(test_in), 'output': np.array(test_out)}}


def choose_samples(data_x, data_y, number):
    """
    This funtion chooses 'number' samples from data_x and data_y, and returns a pair of matrices in the form
    of a dictionary {'input':<...>, 'output':<...>}. This is meant to be used to choose a validation set from
    a subset of the data which might have been set aside for validation purposes.
    """
    test_indices = set(list(np.random.choice(len(data_x), size = [number], replace = False)))
    test_in   = []
    test_out  = []
    for index, point in enumerate(data_x):
        if index in test_indices:
            test_in.append(data_x[index])
            test_out.append(data_y[index])
    return {'input': np.array(test_in), 'output': np.array(test_out)}


def simple_generator(data_x, data_y, batch_size, epochs, validation = None, validation_size = None):
    """
    This simple genrator takes in an array of inputs, and an array of outputs, and splits them
    into batches until all the data has been seen 'epochs' many times.  If 'validation' is set,
    a portion of the data will be set aside for validation purposes, and at every batch, a
    'validation_size' many samples from that validation set will be extracted. and sent to the training
    loop for testing.  If 'validation' is set and 'validation_size' is not, then the entire test set will
    be used for validation at every batch.
    """
    N                 = len(data_x)
    total             = N * epochs
    total_num_batches = total // batch_size
    batches_per_epoch = len(data_x) // batch_size
    validation_split  = basic_train_test_split(data_x, data_y, validation if validation is not None else 0)
    data_x = validation_split['train']['input']
    data_y = validation_split['train']['output']
    test_x = validation_split['test']['input']
    test_y = validation_split['test']['output']
    I      = 0
    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            b_x = data_x[:batch_size]
            b_y = data_y[:batch_size]
            data_x = np.roll(data_x, -batch_size, axis = 0)
            data_y = np.roll(data_y, -batch_size, axis = 0)
            if validation_size is None:
                validation_size = 100
            num_validation = min(validation_size, len(test_x))
            validation_data = choose_samples(test_x, test_y, num_validation) #int(batch_size * validation))
            I += 1
            yield {'train_x':  b_x,
                   'train_y':  b_y,
                   'validate': {'in':  validation_data['input'],
                                'out': validation_data['output']} if validation is not None else None,
                   'batch_number':  batch,
                   'epoch_number':  epoch,
                   'batch_index':   I,
                   'total_batches': total_num_batches,
                   'total_epochs':  epochs}


max_line_length = 0

def basic_callback(**args):
    global max_line_length
    remaining_time_est = args['batch_time'] * (args['total_batches'] - args['batch_index'])
    line = "\rEpoch {0} of {1} --- A: {2:.2f}  - L: {3:.2f} --- VA: {4:.2f}  - VL: {5:.2f}--- Remaining time: {6}"
    line = line.format(args['epoch_number'] + 1, args['total_epochs'],
                       args['batch_accuracy'], args['batch_loss'],
                       args['validation_accuracy'], args['validation_loss'],
                       '{0:02d}:{1:02d}'.format(int(remaining_time_est) // 60, int(remaining_time_est) % 60),
                       )
    if len(line) <= max_line_length:
        line += " "*(len(line) - max_line_length + 1)
        max_line_length = len(line)
    sys.stdout.write(line)
    sys.stdout.flush()

def train(model, data, loss = None, optimizer = None, initial_weights = None,
          callbacks = None, checkpoint_interval = None, checkpoint_folder = None,
          model_weight_file = None):
    """
    Train the model 'model' on the data generated by 'data'. If provided, 'initial_weights' should
    be the name of a file produced by model.save_weights. The model will be initialized using model.load_weights
    before training.

    model: the model to be trained.  The model will be compiled with the 'loss' and 'optimizer' given as parameters
        before training starts.
    data: a GENERATOR, which yields data_points of the format:
           {'train_x':<...>,
            'train_y':<...>,
            'validate': None or {'in':  <...>,
                                 'out': <...>}
            'batch_number':  0,
            'epoch_number':  0,
            'total_batches': 0,
            'total_epochs':  0
           }
    initial_weights: resume the training of a model by preloading the weights from a previous checkpoint_interval
    callbacks: a list of functions that will be called after each batch. Each function is called with
                    (batch_number, epoch_number, loss, accuracy)
    (batch_size, epochs): the size of evety batch and the number of epochs for the training. The functions
                    'generate_data' will generate inputs in chunks of size 'batch_size' until each element of
                    the training set has been seen 'epochs' many times.
    checkpoint_interval: save a copy of the weights after each of these intervals
    """

    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])
    if initial_weights is not None and os.path.exists(initial_weights):
        try:
            model.load_weights(initial_weights)
        except:
            pass
    B = 0
    for data_point in data:
        t1 = time.time()
        loss, accuracy = model.train_on_batch(data_point['train_x'], data_point['train_y'])
        validation_loss = validation_accuracy = 0
        if data_point['validate'] is not None:
            validation_loss, validation_accuracy = model.test_on_batch(data_point['validate']['in'], data_point['validate']['out'])

        if checkpoint_interval is not None and B % checkpoint_interval == 0:
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            w_file_name = 'checkpoint-epoch-{0}-batch-{1}-loss-{2}-accuracy-{3}.hd5'
            w_file_name = w_file_name.format(data_point['epoch_number'],
                                             data_point['batch_number'],
                                             loss,
                                             accuracy)
            weight_file_name = os.path.join(checkpoint_folder, w_file_name)
            model.save_weights(weight_file_name)
        batch_time = time.time() - t1
        for c in (callbacks if callbacks is not None else []):
            try:
                c(model               = model,
                  batch_number        = data_point['batch_number'],
                  batch_index         = data_point['batch_index'],
                  epoch_number        = data_point['epoch_number'],
                  total_batches       = data_point['total_batches'],
                  total_epochs        = data_point['total_epochs'],
                  batch_time          = batch_time,
                  batch_loss          = loss,
                  batch_accuracy      = accuracy,
                  validation_loss     = validation_loss,
                  validation_accuracy = validation_accuracy)
            except Exception as e:
                # DO SOMETHING
                print('FOO', e)
                pass
        B += 1
    # This is the end of the training loop.  We should save the model weights
    save_dir = os.path.dirname(model_weight_file)
    if save_dir != '' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_weights(model_weight_file)
