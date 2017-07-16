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
        output_ = io.StringIO()
        output_.write('<html><body>')
        output_.write('<style>{s}</style>'.format(s=open('notify/style.css').read()))

        rows = []
        colors = []
        for text, true, predicted in d['output']:
            sentiments = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
            color = {True: 'rgba(0, 255, 0, 0.3);', False: 'rgba(255, 0, 0, 0.3);'}[true == predicted]
            true = sentiments[true]
            predicted = sentiments[predicted]
            rows.append([text, true, predicted])
            colors.append(color)

        time = """<table><tr><td>Epoch {epoch_no} of {num_epochs} ({epoch_progress:.2f}%) complete</td>
        <td>ELAPSED TIME:</td>
        <td>REMAINING TIME:</td></tr><tr><td></td><td>{elapsed_time}</td><td>{remaining_time}</td></tr></table>"""
        time = time.format(epoch_no=self._epoch_number,
                           num_epochs=self._total_epochs,
                           epoch_progress=self.epoch_percent,
                           elapsed_time=self.elapsed_time,
                           remaining_time=self.remaining_time)
        output_.write(time)
        output_.write("<p></p>")
        output_.write("<p></p>")

        accuracy = d['accuracy']
        loss = d['loss']
        stats = """<table><tr><td><b>LOSS:</b> {loss:.4f}</td><td style='text-align: right'><b>ACCURACY:</b> {accuracy:.2f}%</td></tr></table>"""
        stats = stats.format(loss=loss, accuracy=100 * accuracy)
        output_.write(stats)

        output_.write("<p></p>")
        output_.write(format_confusion_matrix({'NEGATIVE': 'NEGATIVE',
                                               'NEUTRAL': 'NEUTRAL',
                                               'POSITIVE': 'POSITIVE'},
                                              [x[1] for x in rows],
                                              [x[2] for x in rows]
                                              ))
        output_.write("<p></p>")
        output_.write(format_table(rows, ['Text', 'Truth', 'Predicted'], ['left', 'right', 'right'], sizes=[70, 15, 15], row_colors=colors))
        output_.write('</body></html>')
        ou = open('foo.html', 'w')
        ou.write(output_.getvalue())
        return d

    def pad(self, array, length):
        array = list(array[:length])
        array += [0] * (length - len(array))
        return array


    #def start_threaded_training

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


#start_training()
