import numpy as np
import struct
#from models.utils import train_test_split_indices



TWEET_DATA_FILE  = 'datasets/binary_twitter_training_set.db'
TWEET_INDEX_FILE = 'datasets/binary_twitter_index_set.db'


twitter_index = open(TWEET_INDEX_FILE, 'rb').read()
tweets = open(TWEET_DATA_FILE, 'rb')
size = struct.calcsize('=III')
offset = 0
index = {}
print('READING INDEX')
while offset + size < len(twitter_index):
    tw_index, tw_offset, tw_length =  struct.unpack_from('=III', twitter_index, offset)
    index[tw_index] = {'offset': tw_offset, 'length':tw_length}
    #print(tw_index, tw_offset, tw_length)
    offset += size
print('DONE')

test_size          = int(0.15 * len(index))
test_index_numbers = set(list(np.random.choice(len(index), size = [test_size], replace = False)))

train_indices = {idx:index[idx] for idx in index if idx not in test_index_numbers}
test_indices  = {idx:index[idx] for idx in test_index_numbers}

print(test_size, len(index), len(train_indices), len(test_indices))


def pad(array, length):
    array = list(array[:length])
    array += [0]*(length - len(array))
    return array

def training_batches(batch_size, epochs, validation_size = None):
    N                 = len(train_indices)
    total             = N * epochs
    total_num_batches = total // batch_size
    batches_per_epoch = N // batch_size
    validation_size   = int(batch_size * validation_size) if validation_size is not None else 0
    train_size        = batch_size - validation_size

    I = 0
    indices = list(train_indices.keys())
    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            batch_indices = indices[:batch_size]
            offset = 0
            batch_x = []
            batch_y  = []
            validation_x = []
            validation_y = []
            for i in batch_indices[:train_size]:
                offset = train_indices[i]['offset']
                length = train_indices[i]['length']
                tweets.seek(offset)
                tweet_data = tweets.read(length)
                sentiment = list(tweet_data[0:2])
                tweet_data = tweet_data[2:]
                batch_x.append(pad(tweet_data, 560))
                batch_y.append(sentiment)
                #offset += length

            for i in batch_indices[train_size:]:
                offset = train_indices[i]['offset']
                length = train_indices[i]['length']
                tweets.seek(offset)
                tweet_data = tweets.read(length)
                sentiment = list(tweet_data[0:2])
                tweet_data = tweet_data[2:]
                validation_x.append(pad(tweet_data, 560))
                validation_y.append(sentiment)

            indices = np.roll(indices, -batch_size, axis = 0)
            b_x = np.array(batch_x)
            b_y = np.array(batch_y)
            v_x = np.array(validation_x)
            v_y = np.array(validation_y)
            I += 1
            #print(b_x.shape, b_y.shape)
            yield {'train_x':  b_x,
                   'train_y':  b_y,
                   'validate_x': v_x,
                   'validate_y': v_y,
                   'batch_number':  batch,
                   'epoch_number':  epoch,
                   'batch_index':   I,
                   'total_batches': total_num_batches,
                   'total_epochs':  epochs}


#for x in training_batches(2000, 10, 0.1):
#    print(x['batch_index'], x['total_batches'])
