from twitter_sentiment.convolutional_model_1 import model
import glob
import numpy as np
from twitter_sentiment.buzz_sentiment_analyze import *


model.load_weights('data/convolutional_character_model.hd5')

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
vec_decode = [1,0]


test_in  = []
test_out = []
for i, point in enumerate(read_file('twitter_sentiment/datasets/testing_tweets.csv')):
    if len(point['text']) < 144:
        test_in.append(point['text'] + [0]*(144-len(point['text'])))
    else:
        test_in.append(point['text'][0:144])
    test_out.append(vec_code[point['sentiment']])

#data_in  = np.array(data_in)
#data_out = np.array(data_out)
test_in_array  = np.array(test_in)
test_out_array = np.array(test_out)

foo = model.predict(test_in_array)

N   = len(foo)
#IND = 29

C = 0
for i, x in enumerate(foo):
    #print(x, np.argmax(x), '... true value...', np.argmax(test_out[i]))
    if  vec_decode[np.argmax(x)] == vec_decode[np.argmax(test_out[i])]:
        C += 1
print(float(C) / i)
#print (foo.shape)

print("        ", vec_decode)
for x in vec_decode:
    nums  = {x:0 for x in vec_decode}
    wrong = {x:0 for x in vec_decode}

    #for y in vec_decode:
    #wrong = 0
    #print(wrong)
    for idx, predicted_value in enumerate(foo):
        pred_sent = vec_decode[np.argmax(predicted_value)]
        real_sent = vec_decode[np.argmax(test_out_array[idx])]
        if real_sent == x:
            nums[pred_sent] += 1
            others = [y for y in vec_decode if y != x]
            if pred_sent in others:
                for z in vec_decode:
                    wrong[z] += 1
    #print(wrong)
    print(x, [nums[x] for x in vec_decode], float(wrong[pred_sent])/ sum([nums[x] for x in vec_decode]))


for index, tweet in enumerate(test_in):
    tweet_text = ''.join([chr(x) for x in tweet if x != 0])
    vader = analyze_sentiment_vader_lexicon(tweet_text)
    senti = analyze_sentiment_sentiwordnet_lexicon(tweet_text)
    neuro = vec_decode[np.argmax(foo[index])]
    truth = vec_decode[np.argmax(test_out[index])]
    print(index, "{:*<60}".format(tweet_text[:55]), truth, neuro, vader, senti)
    #print(len(tweet_text))
