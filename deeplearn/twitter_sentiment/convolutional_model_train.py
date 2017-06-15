from convolutional_model_1 import model
import glob
import numpy as np
import time
import sys
training_data_folder = 'twitter_sentiment/batch_files/*.csv'
training_data_files  = [path for path in glob.glob(training_data_folder)]

#print(training_data_files)
# each training data file contains a batch of about 1000 tweets. For the convolutional network
# we leave the batch as is. First we choose 10% of the files at random to serve as a training set,
# and 2% of the files to serve as a test set (because the dataset is so huge). As a first approximation
# use the simple_generator defined in the models module.

from batch_generator import training_batches
import sys

max_line_length = 0

CHECKPOINT     = 'datasets/very-deep-cnn-checkpoint.hd5'
MODEL_FILENAME = 'datasets/very-deep-cnn.hd5'

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

batch_iterator = training_batches(batch_size = 2000, epochs = 250, validation_size = 0.1)

def train(model, data, loss = None, accuracy = None, optimizer = None, initial_weights = None,
          checkpoint_interval = None, checkpoint_folder = None,
          model_weight_file = None):
    model.compile(loss = loss, optimizer = optimizer, metrics = accuracy if accuracy is not None else ['accuracy'])
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
            validation_loss, validation_accuracy = model.test_on_batch(data_point['validate_x'], data_point['validate_y'])

            batch_time = time.time() - t1
            basic_callback(model               = model,
                           data_point          = data_point,
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
            B += 1
        model.save_weights(MODEL_FILENAME)
    except KeyboardInterrupt:
        print()
        print('Stopping')
        print('Saving the weights')
        model.save_weights(CHECKPOINT)
        sys.exit(0)


train(model,
      batch_iterator,
      loss                = 'binary_crossentropy',
      optimizer           = 'rmsprop',
      #callbacks           = [basic_callback],
      checkpoint_interval = 50,
      checkpoint_folder   = 'data/checkpoints',
      model_weight_file   = 'data/test_mod_save_func.hd5')



print("Training done!!!")
print("Writing the model's weights to 'data/convolutional_character_model.hd5'")
model.save_weights('data/convolutional_character_model.hd5')
