from models.rnn_classifier.simple_gru import SimpleGRUClassifier, SimpleGRUClassifierConv
import tensorflow as tf
import numpy as np
import sys
import signal
from train.supervisor import TrainingSupervisor
from train.data import sentiment_training_generator, cms_training_generator
import argparse
from config import stream
import tensorflow as tf
from models.tf_session import tf_session

flags = argparse.ArgumentParser()
stream.fill_arg_parser(flags)
flags.add_argument('-i', '--train-table', dest='train_table', type=str, default='', help='The training input table')
flags.add_argument('-n', '--min-length',  dest='length_cutoff', type=int, default=10, help='The minimum length of strings to send to the training server')
flags.add_argument('-m', '--max-length',  dest='max_length', type=int, default=140, help='The maximum length of a tweet to send to the training server')
flags = flags.parse_args()

supervisor = None


def start_training():
    global supervisor
    model = SimpleGRUClassifier()
    model.build_training_model()
    model.initialize()
    tf_session().run(tf.global_variables_initializer())
    foo = TrainingSupervisor(model, flags.validation_interval)
    supervisor = foo
    data_generator = sentiment_training_generator(batch_size=flags.batch_size, epochs=flags.epochs, validation_size=flags.validation_size)

    try:
        foo.run_training(data_generator['train'], data_generator['validation'])  # (batch_generator, validation_iterator)
    except KeyboardInterrupt:
        save_before_exiting()
        foo.shutdown()
        sys.exit(0)
    print('done')


if __name__ == '__main__':
    start_training()
