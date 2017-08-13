#from twitter_sentiment.convolutional_model_1 import model
import glob
import numpy as np
import keras
import string
#import numpy as np
#training_data_folder = 'twitter_sentiment/batch_files/*.csv'
#training_data_files  = [path for path in glob.glob(training_data_folder)]

#print(training_data_files)
# each training data file contains a batch of about 1000 tweets. For the convolutional network
# we leave the batch as is. First we choose 10% of the files at random to serve as a training set,
# and 2% of the files to serve as a test set (because the dataset is so huge). As a first approximation
# use the simple_generator defined in the models module.

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import Dense, Input, concatenate, Lambda
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Concatenate
from keras import losses
from keras import metrics
from keras import backend as K




from models.batch_generator import simple_generator
from models.train import train
from models.progress_display import basic_callback

from text_generator import rnn_model
from text_generator.read_data import read_data_files_from_folder
from text_generator.batch_generator import rnn_minibatch_sequencer
from text_generator.rnn_model import model, loss_function, SEQ_LEN, BATCH_SIZE, ALPHABET_SIZE, INTERNAL_SIZE
#from models.utils import train_test_split_indices


#training_files      = [training_data_files[i] for i in train_test_split_indices(len(training_data_files), 0.1)]
#training_data_files = [x for x in training_data_files if x not in training_files]
#test_files          = [training_data_files[i] for i in train_test_split_indices(len(training_data_files), 0.01)]

#print (len(training_files))
#print (len(test_files))
#import os
#print(os.getcwd())
#print (__file__)
#def read_file(file):
#    index = 0
##    for f in files:
##        p = f
####        sentiment, text = line[:-1].split('\t')
#            if text.lower().strip() != 'not available':
#                yield {'sentiment':int(sentiment), 'text':[min(ord(c), 255) for c in text]}
#                index += 1
#        except:
#            print(line)
#
#vec_code = {1: [1,0],
#            0: [0,1]}
##            'neutral':  [0,0,1]}''
#vec_decode = [1,0]
#

text_data, validation_data, book_ranges = read_data_files_from_folder('text_generator/training_data/harry_potter/*.txt')
text_data = [ord(c) for c in text_data]


#data_in  = []
#data_out = []
#for i, point in enumerate(read_file('twitter_sentiment/datasets/training-tweets.csv')):
#    if len(point['text']) < 144:
#        data_in.append(point['text'] + [0]*(144-len(point['text'])))
#    else:
#        data_in.append(point['text'][0:144])
#    data_out.append(vec_code[point['sentiment']])

def make_inference_model():
    BATCH_SIZE = 1
    SEQ_LEN = 1
#    if batch_index % 20 == 0:
    input_         = Input(shape = [SEQ_LEN], batch_shape = [BATCH_SIZE, SEQ_LEN], dtype='uint8') #[BATCH_SIZE, SEQ_LEN]
    input_one_hot  = Lambda(K.one_hot, arguments={'num_classes': ALPHABET_SIZE}, output_shape=[SEQ_LEN, ALPHABET_SIZE])(input_)


    lay = GRU(INTERNAL_SIZE, batch_input_shape = (BATCH_SIZE, SEQ_LEN, ALPHABET_SIZE), activation = 'relu', use_bias=True, dropout=0.2, recurrent_dropout=0.2, stateful = True, return_sequences = True)(input_one_hot)
    lay = GRU(INTERNAL_SIZE, batch_input_shape = (BATCH_SIZE, SEQ_LEN, ALPHABET_SIZE), activation = 'relu', use_bias=True, dropout=0.2, recurrent_dropout=0.2, stateful = True, return_sequences = True)(lay)
    lay = GRU(INTERNAL_SIZE, batch_input_shape = (BATCH_SIZE, SEQ_LEN, ALPHABET_SIZE), activation = 'relu', use_bias=True, dropout=0.2, recurrent_dropout=0.2, stateful = True, return_sequences = True)(lay)
    predictions = TimeDistributed(Dense(ALPHABET_SIZE, activation='softmax'))(lay)

    output_ = Input(shape = [SEQ_LEN], batch_shape = [BATCH_SIZE, SEQ_LEN], dtype='uint8') #[BATCH_SIZE, SEQ_LEN]
    output_one_hot  = Lambda(K.one_hot, arguments={'num_classes': ALPHABET_SIZE}, output_shape=[SEQ_LEN, ALPHABET_SIZE])(output_)


    model_output = keras.layers.concatenate([output_one_hot, predictions], axis = 1)
    #loss = losses.categorical_crossentropy(output_one_hot, predictions)
    return Model(input = [input_, output_], outputs = model_output)

inference_model = make_inference_model()



def generate_text(model, num_lines, max_chars_ler_line, **args):
    #seed = [" "*SEQ_LEN]
    #seex.extend()
    global inference_model
    BATCH_SIZE = 1
    SEQ_LEN = 1
    inference_model.set_weights(model.get_weights())

    generated_lines = []

    generated_text = ''
    character  = np.array([ord('Z')])
    while len(generated_lines) < num_lines:
        nextCharProbs = inference_model.predict([character, np.zeros(shape = [1,1])])[:, SEQ_LEN:, :]
        nextCharProbs = np.asarray(nextCharProbs).astype('float64') # Weird type cast issues if not doing this.
        nextCharProbs = nextCharProbs / nextCharProbs.sum()         # Re-normalize for float64 to make exactly 1.0.

        nextCharId = np.random.multinomial(1, nextCharProbs.squeeze(), 1).argmax()
        nextCharId = nextCharId if chr(nextCharId) in string.printable else ord(" ")

        char = chr(nextCharId)
        if char in '\n':
            char_l = '{0: <%s}'%max_chars_ler_line
            generated_lines.append(char_l.format(generated_text))
            generated_text = ''
        elif char in '\r\f\v\b':
            continue
        else:
            generated_text += chr(nextCharId)

        if len(generated_text) == max_chars_ler_line:
            generated_lines.append(generated_text)
            generated_text = ''

        character = np.array([nextCharId])
    return generated_lines

def test_callback(model, batch_index, data_point,  **args):
    # Every 20 batches, print the predicted values:
    if batch_index % 250 == 0:
        inputs = data_point['train_x'][0]
        outputs = data_point['train_x'][1]
        predictions = model.predict_on_batch(data_point['train_x'])
        predictions = predictions[:, SEQ_LEN:, :]
        generated_lines = generate_text(model, inputs.shape[0], 130)
        print()
        print()
        p_line = "| {0:^30} | {1:^30} | {2:^30} | {3:^130} |".format('INPUT', 'OUTPUT', 'PREDICTION', 'GENERATED TEXT',)
        print(p_line)
        for index, line in enumerate(predictions):
            input_  = "".join([chr(x) for x in inputs[index]])
            output_ = "".join([chr(x) for x in outputs[index]])
            pred_   = "".join([chr(np.argmax(x)) for x in line])
            p_line = "| {0:>30} | {1:>30} | {2:>30} | {3:<130} |".format(input_.replace('\n', ' '), output_.replace('\n', ' '), pred_.replace('\n', ' '), generated_lines[index])
            print(p_line)


batch_iterator = rnn_minibatch_sequencer(text_data, BATCH_SIZE, SEQ_LEN, 100)

train(model,
      batch_iterator,
      loss                = rnn_model.loss_function,
      accuracy            = [rnn_model.accuracy],
      optimizer           = 'adam',
      callbacks           = [basic_callback, test_callback],
      checkpoint_interval = 50,
      checkpoint_folder   = 'data/checkpoints',
      model_weight_file   = 'data/test_mod_save_func.hd5')

print("Training done!!!")
print("Writing the model's weights to 'data/convolutional_character_model.hd5'")
model.save_weights('data/convolutional_character_model.hd5')
