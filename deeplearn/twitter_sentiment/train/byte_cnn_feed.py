import zmq
import argparse
import pymysql
import numpy as np
from stream import DataStreamer


flags = argparse.ArgumentParser()
flags.add_argument('-d', '--database', dest='database', type=str, default='', help='The training database')###
flags.add_argument('-H', '--host', dest='host', type=str, default='127.0.0.1', help='The training database IP')###
flags.add_argument('-u', '--user', dest='user', type=str, default='root', help='The training database username')###
flags.add_argument('-p', '--password', dest='password', type=str, default='', help='The training database password')###
flags.add_argument('-i', '--train-table', dest='train_table', type=str, default='', help='The training input table')
flags.add_argument('-s', '--stream-to', dest='stream_to', type=str, default='localhost:6969', help='The IP address of the training loop to send the data to, in the format 0.0.0.0:port')###
flags.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=75, help='The size of the mini-batches to sent to the training server')###
flags.add_argument('-V', '--validation-size', dest='validation_size', type=int, default=25, help='The number of validation samples to send to the training server')###
flags.add_argument('-I', '--validation-interval', dest='validation_interval', type=int, default=15, help='Validate every N batches')###
flags.add_argument('-e', '--epochs', type=int, dest='epochs', default=5, help='The number of epochs, i.e. the number of times the loop should see each elmenet of the data set')###
flags.add_argument('-c', '--min-length',  dest='length_cutoff', type=int, default=10, help='The minimum length of strings to send to the training server')
flags.add_argument('-m', '--max-length',  dest='max_length', type=int, default=140, help='The maximum length of a tweet to send to the training server')
flags = flags.parse_args()

#'sentiment_analysis_data'
connection = pymysql.connect(host=flags.host,
                             user=flags.user,
                             password=flags.password,
                             db=flags.database,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

def get_batch(cursor_, data_set='train', batch_size=100, starting_id=0, record_count=None):
    with connection.cursor() as cursor:
        data = []
        remaining = batch_size
        while remaining > 0:
            ids = ','.join([str(i) for i in range(starting_id, starting_id + remaining)])

            sql = """SELECT id, sentiment, sanitized_text
                     FROM twitter_binary_classification
                     WHERE test_row={data_set} AND id BETWEEN {start_id} AND {end_id}"""
            sql = sql.format(data_set={'train': 0, 'test': 1}[data_set],  # id_list = ids)
                             start_id=starting_id, end_id=starting_id + remaining)
            cursor.execute(sql)
            query_data = cursor.fetchall()
            if len(query_data) == 0:
                starting_id += remaining
                starting_id %= record_count
            max_id = max([x['id'] for x in query_data])
            data.extend(query_data)
            starting_id = max_id
            remaining -= len(query_data)

        data = data[:batch_size]
        max_id = max([x['id'] for x in data])
        batch = []
        for line in data:
            tweet = line['sanitized_text']
            bytes_ = [ord(x) for x in tweet if 0 < ord(x) < 256]
            batch.append({'sanitized_text': bytes_, 'sentiment': line['sentiment']})
        batch_x = [element['sanitized_text'] for element in batch]
        batch_y = [[element['sentiment']] for element in batch]
        return [max_id, batch_x, batch_y]


def get_ids(self, dataset=0):
    with connection.cursor() as cursor:
        c = "SELECT id from twitter_binary_classification WHERE test_row={dataset}"
        c = c.format(dataset=dataset)
        cursor.execute(c)
        ids = [x['id'] for x in cursor.fetchall()]
        return ids

def generate_batches(data_set = 'train', batch_size = 10, epochs = None):
    with connection.cursor() as cursor:
        c = "SELECT COUNT(id) as N from twitter_binary_classification WHERE test_row={data_set}"
        c = c.format(data_set={'train':0, 'test':1}[data_set])
        cursor.execute(c)
        N = cursor.fetchone()['N']
        total = None
        total_num_batches = None
        if epochs is not None:
            total = N * epochs
            total_num_batches = total // batch_size
        batches_per_epoch = N // batch_size
        I = 0
        epoch = 1
        while True:
            offset = 0
            validation_offset = 0
            for batch in range(batches_per_epoch):
                o, batch_x, batch_y = get_batch(cursor, 'train', starting_id=offset, batch_size=batch_size, record_count=N)
                I += 1
                yield {'train_x':  batch_x,
                       'train_y':  batch_y,
                       'batch_number':  batch,
                       'epoch_number':  epoch,
                       'batch_index':   I,
                       'total_batches': total_num_batches,
                       'total_epochs':  epochs}
                offset = o
            epoch += 1

host, port = flags.stream_to.split(':')
port = int(port)
streamer = DataStreamer(host=host, port=port)

batch_generator = generate_batches(data_set = 'train', batch_size=flags.batch_size, epochs=flags.epochs)
validation_iterator = generate_batches(data_set = 'test', batch_size=flags.validation_size, epochs = None)

for x in batch_generator: #generate_batches(batch_size=flags.batch_size, epochs=flags.epochs, validation_size=flags.validation_size):
    print (len(x['train_x']), len(x['train_y']), x['batch_number'], x['epoch_number'], x['batch_index'], x['total_batches'])
    streamer.send({'action':'train',
                   'payload':{'train_x':x['train_x'],
                              'train_y':x['train_y']}})
    if x['batch_index'] % flags.validation_interval == 0:
        validation_data = next(validation_iterator)
        streamer.send({'action':'validate',
                       'payload':{'train_x':validation_data['train_x'],
                                  'train_y':validation_data['train_y']}})
