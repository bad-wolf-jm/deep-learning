import zmq
import argparse
import pymysql
import numpy as np
from config import db, stream
from train.dbi import DBConnection


flags = argparse.ArgumentParser()
db.fill_arg_parser(flags)
stream.fill_arg_parser(flags)
flags.add_argument('-i', '--train-table', dest='train_table', type=str, default='', help='The training input table')
flags.add_argument('-n', '--min-length',  dest='length_cutoff', type=int, default=10, help='The minimum length of strings to send to the training server')
flags.add_argument('-m', '--max-length',  dest='max_length', type=int, default=140, help='The maximum length of a tweet to send to the training server')
flags = flags.parse_args()


db_connection = DBConnection(host=flags.host, user=flags.user, password=flags.password)
db_connection.connect('sentiment_analysis_data')


def count_rows(min_id=0, max_id=None):
    return db_connection.count_rows('trinary_sentiment_dataset', 'shuffle_id', min_id, max_id)


def generate_batches(min_id=0, max_id=None, batch_size=10, epochs=None):
    gen = db_connection.batches('trinary_sentiment_dataset', 'shuffle_id', ['sanitized_text', 'sentiment'], batch_size=batch_size, epochs=epochs)
    sentiment_map = {0: 0, 1: 1, -1: 2}
    for b in iter(gen):
        batch_x = []
        batch_y = []
        for row in b:
            bytes_ = [ord(x) for x in row['sanitized_text'] if 0 < ord(x) < 256]
            batch_x.append(bytes_)
            batch_y.append([sentiment_map[row['sentiment']]])
        yield {'train_x': batch_x,
               'train_y': batch_y,
               'batch_number': gen.current_epoch_batch_number,
               'batches_per_epoch': gen.batches_per_epoch,
               'epoch_number': gen.current_epoch_number,
               'batch_index': gen.current_global_batch_number,
               'total_batches': gen.total_number_of_batches,
               'total_epochs': gen.number_of_epochs}
