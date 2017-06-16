import zipfile
import os
import html
import re
import nltk
import sys

#BATCH_SIZE       = 10000
#LENGTH_CUTOFF    = 10
#MAX_TWEET_LENGTH = 140
#
#file_name    = 'datasets/twitter-binary-sentiment-classification-clean.csv.zip'
#batch_folder = 'batch_files'

#if not os.path.exists(batch_folder):#
#    os.makedirs('batch_files')

#batch_file_name_root = 'twitter-binary-sentiment-batch-{0}.csv'
#
#
#foo = zipfile.ZipFile(file_name)
#
#bar = foo.open('twitter-binary-sentiment-classification-clean.csv')

#print("Counting lines...")
#num_lines = 0
#for line in bar:
#    num_lines += 1
#
#print('Found', num_lines - 1, 'lines in the source file')
##sys.exit()

def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '='):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

#train_file_name = 'datasets/training-tweets.csv'
#test_file_name  = 'datasets/testing-tweets.csv'
##test_file_name  = 'datasets/validation-tweets.csv'
#
#training_size   = 100000
#validation_size = 5000
#t#esting_size    = 25000
#
#buffer_    = []
#index      = 1
#batch_no   = 1
#num_tweets = 0
#
#tweet_lengths   = {}
#word_dictionary = {}
#url_re = '^(?!mailto:)(?:(?:http|https|ftp)://)(?:\\S+(?::\\S*)?@)?(?:(?:(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[0-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))|(?:(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,})))|localhost)(?::\\d{2,5})?(?:(/|\\?|#)[^\\s]*)?$'
#url_re = re.compile(url_re)

#print(url_re.match('http://tinyurl.com'))


import struct

bar = foo.open('twitter-binary-sentiment-classification-clean.csv')
training_file = open(train_file_name, 'wb')
sentiments_stats = {0:0, 1:0}

# The first line can be discarded
line = bar.readline()

list_ = []

tweet_index = 0
stats = {}

TWEET_DATA_FILE  = 'datasets/binary_twitter_training_set.db'
TWEET_INDEX_FILE = 'datasets/binary_twitter_index_set.db'

offset = 0

BINARY_DATA_FILE = open(TWEET_DATA_FILE, 'wb')
BINARY_INDEX_FILE = open(TWEET_INDEX_FILE, 'wb')



while True:
    line = bar.readline()
    if len(line) == 0:
        break
    str_line = line.decode('utf-8')
    sent, tweet = str_line.split('\t')
    tweet = tweet[:-1]
    tweet = html.unescape(tweet)

    #remove the quotation marks at the beginning and end of every tweet
    if tweet[0] == '"' and tweet[-1] == '"':
        while tweet[0] == '"' and tweet[-1] == '"':
            tweet = tweet[1:-1]

    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= MAX_TWEET_LENGTH:

        bytes_  = bytes(tweet.encode('utf-8'))
        sentiment = {0: bytes([1,0]),
                     1: bytes([0,1])}[int(sent)]
        bytes_ = sentiment + bytes_
        BINARY_DATA_FILE.write(bytes_)
        BINARY_INDEX_FILE.write(struct.pack('=III', tweet_index, offset, len(bytes_)))
        tweet_index += 1
        offset += len(bytes_)
        print(tweet_index, offset, bytes_[2:].decode('utf-8'))
#
BINARY_DATA_FILE.close()
BINARY_INDEX_FILE.close()

print('Done making the binary files')

foo = open(TWEET_INDEX_FILE, 'rb').read()
tweets = open(TWEET_DATA_FILE, 'rb')
size = struct.calcsize('=III')
offset = 0
while offset + size < len(foo):
    tw_index, tw_offset, tw_length =  struct.unpack_from('=III', foo, offset)
    tweets.seek(tw_offset)
    data = tweets.read(tw_length)
    print(tw_index, data)
    #print()
    offset += size


#    stats[len(bytes_)] = stats[len(bytes_)] + 1 if len(bytes_) in stats else 1
    # get the words of the tweets, and remove tagged users. All words are stored in a dictionary
    # to use in word embedding.  WE SHOULD USE A REAL WORD PARSER HERE, PUNCTUATION IS MESSING UP
    #words = [w for w in nltk.word_tokenize(tweet) if not w.startswith('@')]
    #tweet = " ".join(words)
#    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= MAX_TWEET_LENGTH:
#        str_line = "\t".join([sent, tweet+'\n'])
#        tweet = tweet.encode('utf-8')
#        print(tweet)
#        list_.append(tweet)
#        #training_file.write(tweet)
#        sentiments_stats[int(sent)] += 1
#print (stats)
