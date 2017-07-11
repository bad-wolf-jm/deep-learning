import zmq
import argparse
import pymysql
import numpy as np
from stream.sender import DataStreamer
from config import db, stream


flags = argparse.ArgumentParser()
db.fill_arg_parser(flags)
stream.fill_arg_parser(flags)
flags.add_argument('-i', '--train-table', dest='train_table', type=str, default='', help='The training input table')
flags.add_argument('-n', '--min-length',  dest='length_cutoff', type=int, default=10, help='The minimum length of strings to send to the training server')
flags.add_argument('-m', '--max-length',  dest='max_length', type=int, default=140, help='The maximum length of a tweet to send to the training server')
flags = flags.parse_args()

connection = pymysql.connect(host=flags.host,
                             user=flags.user,
                             password=flags.password,
                             db=flags.database,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


sentiment_map = {0: 0, 1: 1, -1: 2}


def get_batch(cursor_, batch_size=100, starting_id=0, record_count=None):
    with connection.cursor() as cursor:
        data = []
        remaining = batch_size
        while remaining > 0:
            ids = ','.join([str(i) for i in range(starting_id, starting_id + remaining)])

            sql = """SELECT id, shuffle_id, text
                     FROM byte2vec__training_strings WHERE
                     shuffle_id BETWEEN {start_id} AND {end_id}"""
            sql = sql.format(start_id=starting_id, end_id=starting_id + remaining)
            cursor.execute(sql)
            query_data = cursor.fetchall()
            if len(query_data) == 0:
                starting_id += remaining
                starting_id %= record_count
            max_id = max([x['shuffle_id'] for x in query_data])
            data.extend(query_data)
            starting_id = max_id
            remaining -= len(query_data)

        data = data[:batch_size]
        max_id = max([x['shuffle_id'] for x in data])
        batch = []
        for line in data:
            tweet = line['text']
            bytes_ = [ord(x) for x in tweet if 0 < ord(x) < 256]
            batch.append({'text': bytes_, 'sentiment': bytes_})
        batch_x = [element['text'] for element in batch]
        return [max_id, batch_x, batch_x]

def count_rows(min_id=0, max_id=None):
    with connection.cursor() as cursor:
        if max_id is not None:
            c = "SELECT COUNT(id) as N from trinary_sentiment_dataset WHERE shuffle_id BETWEEN {min_id} AND {max_id}"
        else:
            c = "SELECT COUNT(id) as N from trinary_sentiment_dataset WHERE shuffle_id >= {min_id}"

        c = c.format(min_id=min_id, max_id=max_id)
        cursor.execute(c)
        N = cursor.fetchone()['N']
        return N

def generate_batches(min_id=0, max_id=None, batch_size=10, epochs=None):
    with connection.cursor() as cursor:
        if max_id is not None:
            c = "SELECT COUNT(id) as N from trinary_sentiment_dataset WHERE shuffle_id BETWEEN {min_id} AND {max_id}"
        else:
            c = "SELECT COUNT(id) as N from trinary_sentiment_dataset WHERE shuffle_id >= {min_id}"

        c = c.format(min_id=min_id, max_id=max_id)
        cursor.execute(c)
        N = cursor.fetchone()['N']
        max_id = max_id or N
        total = None
        total_num_batches = None
        if epochs is not None:
            total = N * epochs
            total_num_batches = total // batch_size
        batches_per_epoch = N // batch_size
        I = 0
        epoch = 1
        while True:
            offset = min_id
            validation_offset = 0
            for batch in range(batches_per_epoch):
                o, batch_x, batch_y = get_batch(cursor, starting_id=offset, batch_size=batch_size, record_count=N)
                I += 1
                yield {'train_x':  batch_x,
                       'train_y':  batch_y,
                       'batch_number':  batch,
                       'batches_per_epoch': batches_per_epoch,
                       'epoch_number':  epoch,
                       'batch_index':   I,
                       'total_batches': total_num_batches,
                       'total_epochs':  epochs}
                offset = o
            epoch += 1
