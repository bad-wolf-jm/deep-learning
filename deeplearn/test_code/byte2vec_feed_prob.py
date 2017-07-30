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

with connection.cursor() as cursor:
    sql = "SELECT query_byte, context_byte,probability FROM joint_byte_distribution WHERE probability!= 0"
    cursor.execute(sql)
    joint_byte_data = cursor.fetchall()
    sql = "SELECT byte, probability FROM byte_distribution"
    cursor.execute(sql)
    byte_data = cursor.fetchall()

#foo = {}
#byte_pairs = joint_byte_data.keys()
#probabilities = [joint_byte_data[i] for i, _ in enumerate()
#for x in joint_byte_data:
#    foo[x['query_byte'], x['context_byte']] = 0


def sample_from_distribution(size, distribution):
    sample =  np.random.multinomial(size, [x['probability'] for x in distribution], size=1)
    sample_batch = [None] * size
    shuffle = np.arange(size)
    np.random.shuffle(shuffle)
    start_index = 0
    for index, num_samples in enumerate(sample[0]):
        byte_pair = distribution[index]
        for i in range(start_index, start_index + num_samples):
            sample_batch[shuffle[i]] = byte_pair
        start_index += num_samples
    return sample_batch

def sample_from_noise_distribution(size, distribution):
    query   = sample_from_distribution(size, distribution)
    context = sample_from_distribution(size, distribution)
    samples = []
    for q, c in zip(query, context):
        samples.append({'query_byte': q['byte'], 'context_byte':c['byte']})
    return samples

def generate_batches(batch_size = 10, epochs = None, noise_ratio = 20):
    epoch = 1
    start_id = 0
    I = 0
    noise_size = noise_ratio * batch_size
    while True:
        positive_samples = sample_from_distribution(batch_size, joint_byte_data)
        noise_samples = sample_from_noise_distribution(noise_size, byte_data)
        samples_source = positive_samples + noise_samples
        labels_source = [1]*batch_size + [0]*noise_size
        shuffle = np.random.permutation(len(samples_source))
        samples = []
        labels =[]
        for index in shuffle:
            sample_ = samples_source[index]
            samples.append([sample_['query_byte'], sample_['context_byte']])
            labels.append(labels_source[index])
        yield {'train_x':  samples,
               'train_y':  [[x] for x in labels],
               'batch_number':  0,
               'epoch_number':  epoch,
               'batch_index':   I,
               'total_batches': 0,
               'total_epochs':  0}

host, port = flags.stream_to.split(':')
port = int(port)
streamer = DataStreamer(host=host, port=port)

batch_generator = generate_batches(batch_size=flags.batch_size, epochs=flags.epochs)
validation_iterator = generate_batches(batch_size=flags.validation_size, epochs = None)
for x in batch_generator:
    print (len(x['train_x']), len(x['train_y']), x['batch_number'], x['epoch_number'], x['batch_index'], x['total_batches'])
    streamer.send({'action':'train',
                   'payload':{'train_x':x['train_x'],
                              'train_y':x['train_y']}})
    if x['batch_index'] % flags.validation_interval == 0:
        validation_data = next(validation_iterator)
        streamer.send({'action':'validate',
                       'payload':{'train_x':validation_data['train_x'],
                                  'train_y':validation_data['train_y']}})
