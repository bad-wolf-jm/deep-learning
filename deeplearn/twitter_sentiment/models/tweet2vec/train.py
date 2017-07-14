from models.tweet2vec.lstm_cnn_autoencode import Tweet2Vec_LSTM
import numpy as np
import sys
import signal
from train.supervisor import TrainingSupervisor
from models.tweet2vec.autoencode import generate_batches, flags, count_rows


#host, port = flags.stream_to.split(':')
#print (host, port)
#port = int(port)
#streamer = TrainingDataStreamer(validation_interval=flags.validation_interval, summary_span=None)

N = count_rows()
test = N // 100
batch_generator = generate_batches(min_id=test + 1, max_id=test + 2000, batch_size=flags.batch_size, epochs=500)
validation_iterator = generate_batches(min_id=0, max_id=test, batch_size=flags.validation_size, epochs=None)

max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 64


class TrainAutoencoder(TrainingSupervisor):
    def train_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_y])
        d = self.model.train(batch_x, batch_y)
        print(d)
        return d

    def validation_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_y])
        d = self.model.validate(batch_x, batch_y)
        return d

    def test_model(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_y])
        d = self.model.validate(batch_x, batch_y)
        return d

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array


model = Tweet2Vec_LSTM()
model.build_training_model()
model.initialize()
foo = TrainAutoencoder(model, flags.validation_interval)


def save_before_exiting(*a):
    path = foo.save_model_image()
    foo.shutdown()
    print('\rProcess terminated, model saved as', path)


signal.signal(signal.SIGTERM, save_before_exiting)

try:
    foo.run_training(batch_generator, validation_iterator)
except KeyboardInterrupt:
    save_before_exiting()
    foo.shutdown()
    sys.exit(0)
