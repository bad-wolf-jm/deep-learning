import zipfile
import os
import html

BATCH_SIZE = 1000
LENGTH_CUTOFF = 10

file_name = 'twitter-binary-sentiment-classification-clean.csv.zip'
batch_folder = 'batch_files'

if not os.path.exists(batch_folder):
    os.makedirs('batch_files')

batch_file_name_root = 'twitter-binary-sentiment-batch-{0}.csv'


foo = zipfile.ZipFile(file_name)

bar = foo.open('twitter-binary-sentiment-classification-clean.csv')

buffer_  = []
index    = 1
batch_no = 1

tweet_lengths = {}

for line in bar:
    str_line = line.decode('utf-8')
    if len(buffer_) == BATCH_SIZE:
        batch_file = batch_file_name_root.format(batch_no)
        batch_file = open(os.path.join(batch_folder, batch_file), 'w')
        for l in buffer_:
            batch_file.write(l)
        batch_file.close()
        print('Wrote ', os.path.join(batch_folder, batch_file_name_root.format(batch_no)))
        batch_no += 1
        buffer_ = []

    sent, tweet = str_line.split('\t')
    tweet = tweet[:-1]
    tweet = html.unescape(tweet)

    if tweet[0] == '"' and tweet[-1] == '"':
        while tweet[0] == '"' and tweet[-1] == '"':
            tweet = tweet[1:-1]

    words = [w for w in tweet.split(' ') if not w.startswith('@')]
    tweet = " ".join(words)
    if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 140:
        str_line = "\t".join([sent, tweet+'\n'])
        buffer_.append(str_line)
        if len(tweet) not in tweet_lengths:
            tweet_lengths[len(tweet)] = 0
        tweet_lengths[len(tweet)] += 1
