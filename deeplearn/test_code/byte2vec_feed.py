import zmq
import argparse
import pymysql
import numpy as np
from stream import DataStreamer


flags = argparse.ArgumentParser()
flags.add_argument('-d', '--database',
                   dest='database',
                   type=str,
                   default='',
                   help='The training database')###
flags.add_argument('-H', '--host',
                   dest='host',
                   type=str,
                   default='127.0.0.1',
                   help='The training database IP')###
flags.add_argument('-u', '--user',
                   dest='user',
                   type=str,
                   default='root',
                   help='The training database username')###
flags.add_argument('-p', '--password',
                   dest='password',
                   type=str,
                   default='',
                   help='The training database password')###
flags.add_argument('-i', '--train-table',
                   dest='train_table',
                   type=str,
                   default='',
                   help='The training input table')
flags.add_argument('-s', '--stream-to',
                   dest='stream_to',
                   type=str,
                   default='localhost:6969',
                   help='The IP address of the training loop to send the data to, in the format 0.0.0.0:port')###
flags.add_argument('-b', '--batch-size',
                   dest='batch_size',
                   type=int,
                   default=75,
                   help='The size of the mini-batches to sent to the training server')###
flags.add_argument('-V', '--validation-size',
                   dest='validation_size',
                   type=int,
                   default=25,
                   help='The number of validation samples to send to the training server')###
flags.add_argument('-I', '--validation-interval',
                   dest='validation_interval',
                   type=int,
                   default=15,
                   help='Validate every N batches')###
flags.add_argument('-e', '--epochs',
                   type=int,
                   dest='epochs',
                   default=5,
                   help='The number of epochs, i.e. the number of times the loop should see each elmenet of the data set')###
flags = flags.parse_args()

connection = pymysql.connect(host=flags.host,
                             user=flags.user,
                             password=flags.password,
                             db=flags.database,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

def generate_batches(window_size = 1, batch_size = 10, epochs = None):
    with connection.cursor() as cursor:

        epoch = 1
        while True:
            start_id = 0
            I = 0
            while True:
                c = "SELECT text from twitter_binary_classification WHERE id BETWEEN {start_id} AND {end_id}"
                c = c.format(start_id=start_id, end_id=start_id + 10)
                cursor.execute(c)
                span = 2*window_size + 1
                results = cursor.fetchall()
                if len(results) == 0:
                    break
                batch = []
                for string in [r['text'] for r in results]:
                    print (start_id, string)
                    while len(string) >= span:
                        string_window = string[:span]
                        x_char = string_window[window_size]
                        context_chars = [string_window[i] for i in range(span) if i != window_size]
                        for c in context_chars:
                            batch.append([[ord(x_char)], [ord(c)]])
                        string = string[span:]
                    if len(batch) >= batch_size:
                        b = batch[:batch_size]
                        batch = batch[batch_size:]
                        batch_x = [x[0] for x in b]
                        batch_y = [x[1] for x in b]
                        yield {'train_x':  batch_x,
                               'train_y':  batch_y,
                               'batch_number':  0,
                               'epoch_number':  epoch,
                               'batch_index':   I,
                               'total_batches': 0,
                               'total_epochs':  0}
                        I += 1
                    start_id += 10
            epoch += 1
            if epoch > epochs:
                break

host, port = flags.stream_to.split(':')
port = int(port)
streamer = DataStreamer(host=host, port=port)

batch_generator = generate_batches(batch_size=flags.batch_size, epochs=flags.epochs)
validation_iterator = generate_batches(batch_size=flags.validation_size, epochs = None)
for x in batch_generator:
    #print (len(x['train_x']), len(x['train_y']), x['batch_number'], x['epoch_number'], x['batch_index'], x['total_batches'])
    streamer.send({'action':'train',
                   'payload':{'train_x':x['train_x'],
                              'train_y':x['train_y']}})
    if x['batch_index'] % flags.validation_interval == 0:
        validation_data = next(validation_iterator)
        streamer.send({'action':'validate',
                       'payload':{'train_x':validation_data['train_x'],
                                  'train_y':validation_data['train_y']}})
