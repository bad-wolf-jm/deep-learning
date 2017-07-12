from convolutional_model_1 import model
import os
import glob
import numpy as np
import html
import time
import sys
import zipfile
import signal

#training_data_folder = 'twitter_sentiment/batch_files/*.csv'
#training_data_files  = [path for path in glob.glob(training_data_folder)]

# print(training_data_files)
# each training data file contains a batch of about 1000 tweets. For the convolutional network
# we leave the batch as is. First we choose 10% of the files at random to serve as a training set,
# and 2% of the files to serve as a test set (because the dataset is so huge). As a first approximation
# use the simple_generator defined in the models module.

#from batch_generator import training_batches
import sys


if not os.path.exists('models'):
    os.makedirs('models')

MODEL_WEIGHT_KILLED_FILE = os.path.join('models', 'byte_cnn_resume_weights.hd5')

# def save_model_weights():
#    print()
#    model.save_weights(MODEL_WEIGHT_KILLED_FILE)
#    print("Model weights saved to", MODEL_WEIGHT_KILLED_FILE)


def save_before_exiting(*a):
    print(a)
    print("Process is being killed, I'm saving the weights")
    model.save_weights(MODEL_WEIGHT_KILLED_FILE)
    print("Model weights saved to", MODEL_WEIGHT_KILLED_FILE)
    print("Exiting")
    sys.exit(0)





max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 140

CHECKPOINT = 'datasets/very-deep-cnn-checkpoint.hd5'
MODEL_FILENAME = 'datasets/very-deep-cnn.hd5'


def basic_callback(**args):
    global max_line_length
    remaining_time_est = args['batch_time'] * (args['total_batches'] - args['batch_index'])
    line = "\rEpoch {0} of {1} --- A: {2:.2f}  - L: {3:.2f} --- VA: {4:.2f}  - VL: {5:.2f}--- Remaining time: {6}"
    line = line.format(args['epoch_number'] + 1, args['total_epochs'],
                       args['batch_accuracy'], args['batch_loss'],
                       args['validation_accuracy'], args['validation_loss'],
                       '{0:02d}:{1:02d}'.format(int(remaining_time_est) //
                                                60, int(remaining_time_est) % 60),
                       )
    if len(line) <= max_line_length:
        line += " " * (len(line) - max_line_length + 1)
        max_line_length = len(line)
    sys.stdout.write(line)
    sys.stdout.flush()


DATA_INPUT = []
DATA_OUTPUT = []

file_name = 'datasets/twitter-binary-sentiment-classification-clean.csv.zip'
batch_folder = 'batch_files'

if not os.path.exists(batch_folder):
    os.makedirs('batch_files')

batch_file_name_root = 'twitter-binary-sentiment-batch-{0}.csv'


foo = zipfile.ZipFile(file_name)

bar = foo.open('twitter-binary-sentiment-classification-clean.csv')
#training_file = open(train_file_name, 'wb')
sentiments_stats = {0: 0, 1: 0}

# The first line can be discarded
line = bar.readline()

#list_ = []
sentiment_stats = {0: 0, 1: 0}
tweet_index = 0
#stats = {}

#TWEET_DATA_FILE  = 'datasets/binary_twitter_training_set.db'
#TWEET_INDEX_FILE = 'datasets/binary_twitter_index_set.db'
#
#offset = 0

#BINARY_DATA_FILE = open(TWEET_DATA_FILE, 'wb')
#BINARY_INDEX_FILE = open(TWEET_INDEX_FILE, 'wb')


while True:
    line = bar.readline()
    if len(line) == 0:
        break
    str_line = line.decode('utf-8')
    sent, tweet = str_line.split('\t')
    tweet = tweet[:-1]
    tweet = html.unescape(tweet)

    # remove the quotation marks at the beginning and end of every tweet
    if tweet[0] == '"' and tweet[-1] == '"':
        while tweet[0] == '"' and tweet[-1] == '"':
            tweet = tweet[1:-1]

    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= MAX_TWEET_LENGTH:

        bytes_ = [ord(x) for x in tweet if 0 < ord(x) < 128]  # list(bytes(tweet.encode('utf-8')))
        #print([type(x) for x in bytes_])
        sentiment = {0: [1, 0],
                     1: [0, 1]}[int(sent)]
        DATA_INPUT.append(bytes_)
        DATA_OUTPUT.append(sentiment)
        sentiment_stats[int(sent)] += 1
        #bytes_ = sentiment + bytes_
        # BINARY_DATA_FILE.write(bytes_)
        #BINARY_INDEX_FILE.write(struct.pack('=III', tweet_index, offset, len(bytes_)))
        tweet_index += 1
        # if tweet_index > 1000:
        #    break
        #offset += len(bytes_)
        #print(tweet_index, offset, bytes_[2:].decode('utf-8'))
#
# BINARY_DATA_FILE.close()
# BINARY_INDEX_FILE.close()

print('----- Done making the binary files -----')
print('-- Found {tweets} many tweets'.format(tweets=len(DATA_INPUT)))
print('-- Found {tweets} positive tweets'.format(tweets=sentiment_stats[1]))
print('-- Found {tweets} negative tweets'.format(tweets=sentiment_stats[0]))


train_size = int(0.01 * len(DATA_INPUT))  # len(DATA_INPUT) - test_size
test_size = int(0.10 * train_size)


train_index_numbers = set(list(np.random.choice(len(DATA_INPUT), size=[train_size], replace=False)))
test_index_numbers = set(list(np.random.choice(train_size, size=[test_size], replace=False)))

#train_indices = {idx:index[idx] for idx in index if idx not in test_index_numbers}
#test_indices  = {idx:index[idx] for idx in test_index_numbers}
TRAINING_SET = [(DATA_INPUT[x], DATA_OUTPUT[x])
                for x in train_index_numbers if x not in test_index_numbers]
print('Made the training set')

TESTING_SET = [(DATA_INPUT[x], DATA_OUTPUT[x]) for x in test_index_numbers]
print('Made the test set')

# print(TRAINING_SET)


def pad(array, length):
    array = list(array[:length])
    array += [0] * (length - len(array))
    return array


def training_batches(batch_size, epochs, validation_size=0):
    global TRAINING_SET
    N = len(TRAINING_SET)
    total = N * epochs
    total_num_batches = total // batch_size
    batches_per_epoch = N // batch_size
    I = 0
    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            batch = TRAINING_SET[:batch_size]
            batch_x = np.array([pad(element[0], MAX_TWEET_LENGTH) for element in batch])
            batch_y = np.array([element[1] for element in batch])

            validation_sample = set(list(np.random.choice(
                len(TESTING_SET), size=[validation_size], replace=False)))
            validation_x = np.array([pad(element[0], MAX_TWEET_LENGTH)
                                     for element in [TESTING_SET[i] for i in validation_sample]])
            validation_y = np.array([element[1] for element in [TESTING_SET[i]
                                                                for i in validation_sample]])
            TRAINING_SET = np.roll(TRAINING_SET, -batch_size, axis=0)
            I += 1
            yield {'train_x':  batch_x,
                   'train_y':  batch_y,
                   'validate_x': validation_x,
                   'validate_y': validation_y,
                   'batch_number':  batch,
                   'epoch_number':  epoch,
                   'batch_index':   I,
                   'total_batches': total_num_batches,
                   'total_epochs':  epochs}


batch_iterator = training_batches(batch_size=750, epochs=250, validation_size=25)


def train(model, data, loss=None, accuracy=None, optimizer=None, initial_weights=None,
          checkpoint_interval=None, checkpoint_folder=None,
          model_weight_file=None):
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=accuracy if accuracy is not None else ['accuracy'])
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
            validation_loss, validation_accuracy = model.test_on_batch(
                data_point['validate_x'], data_point['validate_y'])

            batch_time = time.time() - t1
            basic_callback(model=model,
                           data_point=data_point,
                           batch_number=data_point['batch_number'],
                           batch_index=data_point['batch_index'],
                           epoch_number=data_point['epoch_number'],
                           total_batches=data_point['total_batches'],
                           total_epochs=data_point['total_epochs'],
                           batch_time=batch_time,
                           batch_loss=loss,
                           batch_accuracy=accuracy,
                           validation_loss=validation_loss,
                           validation_accuracy=validation_accuracy)
            B += 1
        model.save_weights(MODEL_FILENAME)
    except KeyboardInterrupt:
        save_before_exiting()


train(model,
      batch_iterator,
      loss='categorical_crossentropy',
      optimizer='adadelta',
      checkpoint_interval=50,
      checkpoint_folder='data/checkpoints',
      model_weight_file='data/test_mod_save_func.hd5')

print("Training done!!!")
print("Writing the model's weights to 'data/convolutional_character_model.hd5'")
model.save_weights('data/convolutional_character_model.hd5')
