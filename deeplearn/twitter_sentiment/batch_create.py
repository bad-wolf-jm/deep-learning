import zipfile
import os
import html
import re
import nltk
import sys

BATCH_SIZE    = 10000
LENGTH_CUTOFF = 10

file_name    = 'datasets/twitter-binary-sentiment-classification-clean.csv.zip'
batch_folder = 'batch_files'

if not os.path.exists(batch_folder):
    os.makedirs('batch_files')

batch_file_name_root = 'twitter-binary-sentiment-batch-{0}.csv'


foo = zipfile.ZipFile(file_name)

bar = foo.open('twitter-binary-sentiment-classification-clean.csv')

num_lines = 0
for line in bar:
    num_lines += 1

print('Found', num_lines - 1, 'lines in the source file')


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



word_dictionary = {}

bar = foo.open('twitter-binary-sentiment-classification-clean.csv')
first_line = True
for index, line in enumerate(bar):
    if first_line:
        first_line = False
        continue

    str_line = line.decode('utf-8')
    #print (str_line)
    sent, tweet = str_line.split('\t')
    tweet = tweet[:-1]
    tweet = html.unescape(tweet)

    # get the words of the tweets, and remove tagged users. All words are stored in a dictionary
    # to use in word embedding.  WE SHOULD USE A REAL WORD PARSER HERE, PUNCTUATION IS MESSING UP
    words = [w for w in nltk.word_tokenize(tweet) if not w.startswith('@')]
    for w in words:
        if w not in word_dictionary:
            word_dictionary[w] = 0
        word_dictionary[w] += 1
    if index %1000 == 0:
        print_progress(index, num_lines, "Counting words")
print()

print('Found', len(word_dictionary), 'distinct words')
print('Writing the dictionary to "word_dictionary.csv"')
file_ = open(os.path.join('datasets/word_dictionary.csv'), 'w')
list_ = reversed(sorted([x for x in word_dictionary], key = lambda x:word_dictionary[x]))

for i, w in enumerate(list_):#
    file_.write("%s\t%s\t%s\n"%(i, word_dictionary[w], w))
file_.close()


    # The tweet is added to the current batch.
#    tweet = " ".join(words)
#    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 140:
#        str_line = "\t".join([sent, tweet+'\n'])
#        buffer_.append(str_line)
#        sentiments_stats[int(sent)] += 1
#        num_tweets += 1
#        if len(tweet) not in tweet_lengths:
#            tweet_lengths[len(tweet)] = 0
#        tweet_lengths[len(tweet)] += 1





train_file_name = 'datasets/training-tweets.csv'
test_file_name  = 'datasets/testing_tweets.csv'

training_size = 60000
testing_size  = 15000




buffer_    = []
index      = 1
batch_no   = 1
num_tweets = 0

tweet_lengths   = {}
word_dictionary = {}
#url_re = '^(?!mailto:)(?:(?:http|https|ftp)://)(?:\\S+(?::\\S*)?@)?(?:(?:(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[0-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))|(?:(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,})))|localhost)(?::\\d{2,5})?(?:(/|\\?|#)[^\\s]*)?$'
#url_re = re.compile(url_re)

#print(url_re.match('http://tinyurl.com'))



bar = foo.open('twitter-binary-sentiment-classification-clean.csv')
training_file = open(train_file_name, 'w')
sentiments_stats = {0:0, 1:0}
line = bar.readline()
for number in range(training_size):
    #print (number)
    line = bar.readline()
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
    words = [w for w in nltk.word_tokenize(tweet) if not w.startswith('@')]
    tweet = " ".join(words)
    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 140:
        str_line = "\t".join([sent, tweet+'\n'])
        training_file.write(str_line)
        sentiments_stats[int(sent)] += 1
    print_progress(number, training_size, 'Making training file:')
print()
print('Wrote', training_size, 'lines to the training file', 'pos:{0} -- neg:{1}'.format(sentiments_stats[1], sentiments_stats[0]))


testing_file = open(test_file_name, 'w')
sentiments_stats = {0:0, 1:0}
for number in range(testing_size):
    line = bar.readline()
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
    words = [w for w in nltk.word_tokenize(tweet) if not w.startswith('@')]
    tweet = " ".join(words)
    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 140:
        str_line = "\t".join([sent, tweet+'\n'])
        testing_file.write(str_line)
        sentiments_stats[int(sent)] += 1
    print_progress(number, training_size, 'Making training file:')

print()
print('Wrote', testing_size, 'lines to the testing file', 'pos:{0} -- neg:{1}'.format(sentiments_stats[1], sentiments_stats[0]))







#print()
#print('Batch size:', BATCH_SIZE)
#print('Wrote', batch_no, 'batch files')
#print('Found', num_tweets, 'tweets')
#
#file_ = open(os.path.join(batch_folder, 'word_dictionary.csv'), 'w')
#list_ = reversed(sorted([x for x in word_dictionary], key = lambda x:word_dictionary[x]))
#
#for i, w in enumerate(list_):##
#    file_.write("%s\t%s\t%s\n"%(i, word_dictionary[w], w))
#file_.close()


    #if url_re.search(w):
    #if i < 200000:
    #    print(w, word_dictionary[w])
