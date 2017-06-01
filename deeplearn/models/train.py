# training utilities for keras networks
import os
import sys
import numpy as np
import logging
import time


from models.batch_generator import basic_train_test_split
from models.exceptions import StopTraining, OvertrainWarning



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
    try:
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
                except StopTraining:
                    print("Training stopped by callback")
                    raise
                except OvertrainWarning:
                    print("Warning: possible overtraining")
                    raise StopTraining()
                except Exception as e:
                    # DO SOMETHING
                    print('FOO', e)
                    pass
            B += 1
    except StopTraining:
        pass
    finally:
        save_dir = os.path.dirname(model_weight_file)
        if save_dir != '' and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_weights(model_weight_file)
