import zipfile
import os
import html
import pymysql
import numpy as np

LENGTH_CUTOFF = 10

file_name = 'datasets/twitter-binary-sentiment-classification-clean.csv.zip'
foo = zipfile.ZipFile(file_name)
bar = foo.open('twitter-binary-sentiment-classification-clean.csv')


connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='sentiment_analysis_data',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


def addslashes(s):
    if s == None:
        return ''
    d = {'"': '\\"', "'": "\\'", "\0": "\\\0", "\\": "\\\\"}
    return ''.join(d.get(c, c) for c in s)


def tokenize(text):
    words = text.split(' ')
    sanitized_words = []
    for w in words:
        if w.startswith('@'):
            sanitized_words.append('@<USER>')
        else:
            if w.startswith('http://') or w.startswith('https://'):
                sanitized_words.append('<URL>')
            else:
                sanitized_words.append(w)
    return ' '.join(sanitized_words)

with connection.cursor() as cursor:
    first_line = True
    insert_batch = []
    num_tweets = 0
    for index, line in enumerate(bar):
        if first_line:
            first_line = False
            continue
        str_line = line.decode('utf-8')
        sent, tweet = str_line.split('\t')
        tweet = tweet[:-1]
        tweet = html.unescape(tweet)
        if tweet[0] == '"' and tweet[-1] == '"':
            while tweet[0] == '"' and tweet[-1] == '"':
                tweet = tweet[1:-1]
        if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 140:
            tweet = addslashes(tweet)
            sanitized_tweet = tokenize(tweet)
            sql = "INSERT INTO twitter_binary_classification (id, test_row, sentiment, text, sanitized_text) VALUES ({id}, 0, {sentiment}, '{text}', '{sanitized_tweet}')"
            sql = sql.format(id=index, sentiment=int(sent), text=tweet, sanitized_tweet = sanitized_tweet)
            insert_batch.append(sql)
            print(tweet[:140])
            num_tweets += 1
        if len(insert_batch) > 250000:
            for inst in insert_batch:
                print(inst[:250])
                cursor.execute(inst)
            connection.commit()
            insert_batch = []
        #if index == 1000:
        #    break
    for inst in insert_batch:
        print(inst[:250])
        cursor.execute(inst)
    connection.commit()
    insert_batch = []

    train_size = int(0.9 * num_tweets)
    test_size = num_tweets -  train_size
    test_index_numbers = set(list(np.random.choice(num_tweets, size=[test_size], replace=False)))
    for i in test_index_numbers:
        sql = """UPDATE twitter_binary_classification SET test_row=1 WHERE id={}""".format(i)
        print(sql)
        cursor.execute(sql)
    connection.commit()
