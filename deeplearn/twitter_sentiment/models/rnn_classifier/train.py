from models.rnn_classifier.bidirectional_gru import Tweet2Vec_BiGRU
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

max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 1024


#class TrainRNNClassifier(TrainingSupervisor):
#    def train_step(self, train_x, train_y):
#        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
#        batch_y = np.array([element for element in train_y])
#        d = self.model.train(batch_x, batch_y)
#        print(d)
#        return d
#
#    def validation_step(self, train_x, train_y):
#        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
#        batch_y = np.array([element for element in train_y])
#        d = self.model.validate(batch_x, batch_y)
#        return d
#
#    def test_model(self, train_x, train_y):
#        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
#        batch_y = np.array([element for element in train_y])
#        d = self.model.test(batch_x, batch_y)
#        return d
#
#    def pad(self, array, length):
#        array = list(array[:length])
#        array += [0] * (length - len(array))
#        return array


#def save_before_exiting(*a):
#    path = foo.save_model_image()
#    foo.shutdown()
#    print('\rProcess terminated, model saved as', path)
#
#
#signal.signal(signal.SIGTERM, save_before_exiting)
#supervisor = None


def start_training():
    global supervisor
    model = Tweet2Vec_BiGRU()
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
