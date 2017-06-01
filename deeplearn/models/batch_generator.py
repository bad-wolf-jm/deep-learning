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
