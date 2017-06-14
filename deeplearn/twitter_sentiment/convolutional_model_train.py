from twitter_sentiment.convolutional_model_1 import model
import glob
import numpy as np

training_data_folder = 'twitter_sentiment/batch_files/*.csv'
training_data_files  = [path for path in glob.glob(training_data_folder)]

#print(training_data_files)
# each training data file contains a batch of about 1000 tweets. For the convolutional network
# we leave the batch as is. First we choose 10% of the files at random to serve as a training set,
# and 2% of the files to serve as a test set (because the dataset is so huge). As a first approximation
# use the simple_generator defined in the models module.

from models.batch_generator import simple_generator
from models.train import train
from models.progress_display import basic_callback
from models.utils import train_test_split_indices


#training_files      = [training_data_files[i] for i in train_test_split_indices(len(training_data_files), 0.1)]
#training_data_files = [x for x in training_data_files if x not in training_files]
#test_files          = [training_data_files[i] for i in train_test_split_indices(len(training_data_files), 0.01)]

#print (len(training_files))
#print (len(test_files))
import os
print(os.getcwd())
print (__file__)
def read_file(file):
    index = 0
#    for f in files:
#        p = f
    f = open(file)
    for line in f.readlines():
        try:
            sentiment, text = line[:-1].split('\t')
            if text.lower().strip() != 'not available':
                yield {'sentiment':int(sentiment), 'text':[min(ord(c), 255) for c in text]}
                index += 1
        except:
            print(line)

vec_code = {1: [1,0],
            0: [0,1]}
#            'neutral':  [0,0,1]}''
vec_decode = [1,0]

data_in  = []
data_out = []
for i, point in enumerate(read_file('twitter_sentiment/datasets/training-tweets.csv')):
    if len(point['text']) < 144:
        data_in.append(point['text'] + [0]*(144-len(point['text'])))
    else:
        data_in.append(point['text'][0:144])
    data_out.append(vec_code[point['sentiment']])

#test_in  = []
#test_out = []
#for i, point in enumerate(read_files(test_files)):#
#    if len(point['text']) < 144:
#        test_in.append(point['text'] + [0]*(144-len(point['text'])))
#    else:
#        test_in.append(point['text'][0:144])
#    test_out.append(vec_code[point['sentiment']])


data_in  = np.array(data_in)
data_out = np.array(data_out)
#test_in  = np.array(test_in)
#test_out = np.array(test_out)

batch_iterator = simple_generator(data_in, data_out, batch_size = 200, epochs = 250, validation = 0.1, validation_size = 200)

train(model,
      batch_iterator,
      loss                = 'binary_crossentropy',
      optimizer           = 'rmsprop',
      callbacks           = [basic_callback],
      checkpoint_interval = 50,
      checkpoint_folder   = 'data/checkpoints',
      model_weight_file   = 'data/test_mod_save_func.hd5')

print("Training done!!!")
print("Writing the model's weights to 'data/convolutional_character_model.hd5'")
model.save_weights('data/convolutional_character_model.hd5')
