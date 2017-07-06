#from convolutional_model_1 import model
from byte_cnn import ByteCNN
from tf_session import tf_session
import tensorflow as tf
import os
import glob
import numpy as np
import html
import time
import sys
import zipfile
import signal
import sys
import pymysql

max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 140

#connection = pymysql.connect(host='10.137.11.91',
#                             user='jalbert',
#                             password='gameloft2017',
#                             db='tren_games',
#                             charset='utf8mb4',
#                             cursorclass=pymysql.cursors.DictCursor)

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root',
                             db='sentiment_analysis_data',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

sql = """SELECT sentiment, sanitized_text FROM trinary_sentiment_dataset"""
with connection.cursor() as cursor:
    cursor.execute(sql)
    data = cursor.fetchall()

data_set = []

for line in data:
    tweet = line['sanitized_text']
    tweet = html.unescape(tweet)

    #if tweet[0] == '"' and tweet[-1] == '"':
    #    while tweet[0] == '"' and tweet[-1] == '"':
    #        tweet = tweet[1:-1]

    #if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= MAX_TWEET_LENGTH:
    bytes_ = [ord(x) for x in tweet if 0 < ord(x) < 256]  # list(bytes(tweet.encode('utf-8')))
    data_set.append({'sanitized_text': bytes_, 'sentiment':line['sentiment']+1})

train_size = int(0.9 * len(data_set))
test_size = len(data_set) -  train_size
train_index_numbers = set(list(np.random.choice(len(data_set), size=[train_size], replace=False)))
test_index_numbers = set(list(np.random.choice(train_size, size=[test_size], replace=False)))

TRAINING_SET = [data_set[x] for x in train_index_numbers if x not in test_index_numbers]
TESTING_SET = [data_set[x] for x in test_index_numbers]

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
            batch_x = np.array([pad(element['sanitized_text'], MAX_TWEET_LENGTH) for element in batch])
            batch_y = np.array([[element['sentiment']] for element in batch])

            validation_sample = set(list(np.random.choice(len(TESTING_SET), size=[validation_size], replace=False)))
            validation_x = np.array([pad(element['sanitized_text'], MAX_TWEET_LENGTH) for element in [TESTING_SET[i] for i in validation_sample]])
            validation_y = np.array([[element['sentiment']] for element in [TESTING_SET[i] for i in validation_sample]])
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


def save_before_exiting(*a):
    print(a)
    print("Process is being killed, I'm saving the weights")
    model.save_weights(MODEL_WEIGHT_KILLED_FILE)
    print("Model weights saved to", MODEL_WEIGHT_KILLED_FILE)
    print("Exiting")
    sys.exit(0)

#CHECKPOINT = 'datasets/very-deep-cnn-checkpoint.hd5'
#MODEL_FILENAME = 'datasets/very-deep-cnn.hd5'


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




def pad(array, length):
    array = list(array[:length])
    array += [0] * (length - len(array))
    return array


batch_iterator = training_batches(batch_size=150, epochs=250, validation_size=75)
foo = ByteCNN()
foo.build_training_model()
foo.initialize()


def train(model, data, loss=None, accuracy=None, optimizer=None, initial_weights=None,
          checkpoint_interval=None, checkpoint_folder=None,
          model_weight_file=None):
    B = 0
    try:
        for data_point in data:
            t1 = time.time()
            d = model.train(data_point['train_x'], data_point['train_y'])
            v = model.validate(data_point['validate_x'], data_point['validate_y'])

            batch_time = time.time() - t1
            #model.write_summary(data_point['train_x'], data_point['train_y'])
            basic_callback(model=model,
                           data_point=data_point,
                           batch_number=data_point['batch_number'],
                           batch_index=data_point['batch_index'],
                           epoch_number=data_point['epoch_number'],
                           total_batches=data_point['total_batches'],
                           total_epochs=data_point['total_epochs'],
                           batch_time=batch_time,
                           batch_loss=d['loss'],
                           batch_accuracy=d['accuracy'],
                           validation_loss=v['loss'],
                           validation_accuracy=v['accuracy'])
            B += 1
        # model.save_weights(MODEL_FILENAME)
    except KeyboardInterrupt:
        pass
        # save_before_exiting()


with tf_session() as session:
    #trainer = CategoricalEncoderTrainer(input_dimension=256, output_dimension=8)
    # trainer.build_inference_model()
    session.run(tf.global_variables_initializer())

    train(foo,
          batch_iterator,
          loss='categorical_crossentropy',
          optimizer='adadelta',
          checkpoint_interval=50,
          checkpoint_folder='data/checkpoints',
          model_weight_file='data/test_mod_save_func.hd5')

print("Training done!!!")
print("Writing the model's weights to 'data/convolutional_character_model.hd5'")
# model.save_weights('data/convolutional_character_model.hd5')
