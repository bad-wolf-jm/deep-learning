import zmq
import argparse
import pymysql
import numpy as np
import random
import time
import collections
from stream.sender import DataStreamer
from config import db, stream

flags = argparse.ArgumentParser()
db.fill_arg_parser(flags)
stream.fill_arg_parser(flags)
flags.add_argument('-n', '--noise-ratio',
                   type=int,
                   dest='noise_ratio',
                   default=10,
                   help='The ratio of negative to positive sample pairs')
flags.add_argument('-N', '--num-skips',
                   type=int,
                   dest='num_skips',
                   default=10,
                   help='The number of correct context samples produced for every query byte')
flags = flags.parse_args()

connection = pymysql.connect(host=flags.host,
                             user=flags.user,
                             password=flags.password,
                             db=flags.database,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


byte_frequencies = {}
with connection.cursor() as cursor:
    sql = "SELECT byte, frequency, probability, keep_probability, unigram_probability FROM byte2vec__byte_frequencies"
    cursor.execute(sql)
    byte_data = cursor.fetchall()
    for x in byte_data:
        byte_frequencies[x['byte']] = x
unigram_probabilities = [byte_frequencies[x]['unigram_probability'] for x in range(0, 256)]
unigram_probabilities = [x / sum(unigram_probabilities) for x in unigram_probabilities]
print (sum(unigram_probabilities))


def draw_random_context_bytes(num):
    return np.random.multinomial(num, unigram_probabilities)


def keep_byte(byte):
    keep_probability = byte_frequencies[byte]['keep_probability']
    keep = np.random.multinomial(1, [keep_probability, 1 - keep_probability])
    return (keep[0] == 1)


def shuffle(ar1, ar2):
    perm = np.random.permutation(len(ar1))
    ar1 = [ar1[i] for i in perm]
    ar2 = [ar2[i] for i in perm]
    return ar1, ar2

def generate_windows(epochs=None, window_size=5):
    with connection.cursor() as cursor:
        sql = "SELECT text FROM byte2vec__training_strings LIMIT 100"
        padding = [0] * window_size
        epoch = 1
        while True:
            if epochs is not None and epoch > epochs:
                break
            cursor.execute(sql)
            buf = cursor.fetchmany(100)
            while len(buf) > 0:
                for line in buf:
                    bytes_ = bytes([b for b in line['text'].encode('utf-8') if b != 0 and keep_byte(b)])
                    num_windows = len(bytes_) // window_size
                    left_overs = len(bytes_) % window_size
                    padding_required = window_size - left_overs
                    bytes_ += bytes([0] * padding_required)
                    for start in range(0, len(bytes_), window_size):
                        yield {'epoch': epoch, 'window': bytes_[start:start + window_size]}
                buf = cursor.fetchmany(100)
            epoch += 1


def generate_batches(batch_size=10, epochs=None, noise_ratio=10, window_size=5, num_skips=2):
    assert num_skips <= 2 * window_size
    batch = []
    labels = []
    with connection.cursor() as cursor:
        sql = "SELECT * FROM byte2vec__stats"
        cursor.execute(sql)
        foo = cursor.fetchone()
        num_bytes = foo['num_bytes']

    t_num_bytes = num_bytes * num_skips * (noise_ratio + 1)
    t_num_batches_per_epoch = t_num_bytes // batch_size
    t_num_batches = t_num_batches_per_epoch * epochs if epochs is not None else None
    I = 0
    B = 0
    current_epoch = 0
    for w in generate_windows(epochs, 2 * window_size + 1):
        if current_epoch != w['epoch']:
            current_epoch = w['epoch']
            B = 0
        window = list(w['window'])
        query_byte = window[window_size]
        del window[window_size]
        context_bytes = set([])
        for s in range(num_skips):
            target = random.randint(0, len(window) - 1)
            while target in context_bytes:
                target = random.randint(0, len(window) - 1)
            context_bytes.add(target)
        for c in context_bytes:
            batch.append([query_byte, window[c]])
            labels.append([1])
            random_context = draw_random_context_bytes(noise_ratio)
            for byte, number in enumerate(random_context):
                for _ in range(number):
                    batch.append([query_byte, byte])
                    labels.append([0])
            batch, labels = shuffle(batch, labels)
        while len(batch) >= batch_size:
            I += 1
            B += 1
            yield {'train_x':  batch[:batch_size],
                   'train_y':  labels[:batch_size],
                   'batch_number':  B,
                   'batches_per_epoch': t_num_batches_per_epoch,
                   'epoch_number':  w['epoch'],
                   'batch_index':   I,
                   'total_batches': t_num_batches,
                   'total_epochs':  epochs}
            batch = batch[batch_size:]
            labels = labels[batch_size:]


host, port = flags.stream_to.split(':')
port = int(port)
streamer = DataStreamer(host=host, port=port)

batch_generator = generate_batches(batch_size=flags.batch_size, epochs=flags.epochs, noise_ratio=flags.noise_ratio, num_skips=flags.num_skips)
validation_iterator = generate_batches(batch_size=flags.validation_size, epochs=None, noise_ratio=1, num_skips=2)

batch_train_times = collections.deque(maxlen=500)
validation_times = collections.deque(maxlen=500)
train_travel_times = collections.deque(maxlen=500)


for x in batch_generator:
    print (len(x['train_x']), len(x['train_y']), x['batch_number'], x['epoch_number'], x['batch_index'], x['total_batches'], np.mean(batch_train_times), np.mean(train_travel_times))
    t_0 = time.time()
    vals = streamer.send({'action': 'train', 'payload': {'train_x': x['train_x'], 'train_y': x['train_y']}})
    batch_train_time = time.time() - t_0
    batch_train_times.append(batch_train_time)
    y = vals['return']['time']
    travel_time = batch_train_time - y
    train_travel_times.append(travel_time)
    if x['batch_index'] % flags.validation_interval == 0:
        t_1 = time.time()
        validation_data = next(validation_iterator)
        streamer.send({'action': 'validate', 'payload': {'train_x': validation_data['train_x'], 'train_y': validation_data['train_y']}})
        validation_time = time.time() - t_1
        validation_times.append(batch_train_time)
