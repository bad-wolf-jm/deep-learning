import zipfile
import os
import html
import re
import nltk
import sys

BATCH_SIZE       = 10000
LENGTH_CUTOFF    = 10
MAX_TWEET_LENGTH = 140

file_name    = 'datasets/twitter-binary-sentiment-classification-clean.csv.zip'
batch_folder = 'batch_files'

if not os.path.exists(batch_folder):
    os.makedirs('batch_files')

batch_file_name_root = 'twitter-binary-sentiment-batch-{0}.csv'


foo = zipfile.ZipFile(file_name)

bar = foo.open('twitter-binary-sentiment-classification-clean.csv')

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

train_file_name = 'datasets/training-tweets.csv'
test_file_name  = 'datasets/testing-tweets.csv'
test_file_name  = 'datasets/validation-tweets.csv'

training_size   = 100000
validation_size = 5000
testing_size    = 25000

buffer_    = []
index      = 1
batch_no   = 1
num_tweets = 0

tweet_lengths   = {}
#word_dictionary = {}
#url_re = '^(?!mailto:)(?:(?:http|https|ftp)://)(?:\\S+(?::\\S*)?@)?(?:(?:(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[0-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))|(?:(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,})))|localhost)(?::\\d{2,5})?(?:(/|\\?|#)[^\\s]*)?$'
#url_re = re.compile(url_re)

#print(url_re.match('http://tinyurl.com'))



bar = foo.open('twitter-binary-sentiment-classification-clean.csv')
training_file = open(train_file_name, 'w')
sentiments_stats = {0:0, 1:0}
line = bar.readline()
while True:
    line = bar.readline()
    if len(line) == 0:
        break
    print(line)
    str_line = line.decode('utf-8')
    sent, tweet = str_line.split('\t')
    tweet = tweet[:-1]
    tweet = html.unescape(tweet)

    #remove the quotation marks at the beginning and end of every tweet
    if tweet[0] == '"' and tweet[-1] == '"':
        while tweet[0] == '"' and tweet[-1] == '"':
            tweet = tweet[1:-1]

    # get the words of the tweets, and remove tagged users. All words are stored in a dictionary
    # to use in word embedding.  WE SHOULD USE A REAL WORD PARSER HERE, PUNCTUATION IS MESSING UP
    #words = [w for w in nltk.word_tokenize(tweet) if not w.startswith('@')]
    #tweet = " ".join(words)
    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= MAX_TWEET_LENGTH:
        str_line = "\t".join([sent, tweet+'\n'])
        #training_file.write(str_line)
        sentiments_stats[int(sent)] += 1
