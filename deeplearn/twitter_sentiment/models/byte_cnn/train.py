from models.byte_cnn.byte_cnn import ByteCNN
import numpy as np
import sys
import signal
from train.supervisor import TrainingSupervisor
from models.byte_cnn.sentiment import generate_batches, flags, count_rows
import io
from notify.format import format_table, format_confusion_matrix
from notify.send_mail import EmailNotification

N = count_rows()
test = N // 100
batch_generator = generate_batches(min_id=test + 1, batch_size=flags.batch_size, epochs=flags.epochs)
validation_iterator = generate_batches(min_id=0, max_id=test, batch_size=flags.validation_size, epochs=None)

max_line_length = 0
LENGTH_CUTOFF = 10
MAX_TWEET_LENGTH = 140


class TrainByteCNN(TrainingSupervisor):
    def train_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([element for element in train_y])
        d = self.model.train(batch_x, batch_y)
        print (d)
        return d

    def validation_step(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([element for element in train_y])
        d = self.model.validate(batch_x, batch_y)
        return d

    def test_model(self, train_x, train_y):
        batch_x = np.array([self.pad(element, MAX_TWEET_LENGTH) for element in train_x])
        batch_y = np.array([element for element in train_y])
        d = self.model.test(batch_x, batch_y)
        return d

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array

def save_before_exiting(*a):
    path = foo.save_model_image()
    foo.shutdown()
    print('\rProcess terminated, model saved as', path)


signal.signal(signal.SIGTERM, save_before_exiting)


supervisor = None

def start_training():
    global supervisor
    model = ByteCNN()
    model.build_training_model()
    model.initialize()
    foo = TrainByteCNN(model, flags.validation_interval)
    supervisor = foo
    try:
        foo.run_training(batch_generator, validation_iterator)
    except KeyboardInterrupt:
        save_before_exiting()
        foo.shutdown()
        sys.exit(0)
    print('done')


if __name__ == '__main__':
   start_training()
