import zipfile
import os
import html
import pymysql

LENGTH_CUTOFF = 10

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root',
                             db='sentiment_analysis_data',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

feed_connection = pymysql.connect(host='10.137.11.91',
                             user='jalbert',
                             password='gameloft2017',
                             db='tren_games',
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



sql = """SELECT message, vader, sentiwordnet FROM twitter__tweets WHERE lang='en'"""
with feed_connection.cursor() as feed_cursor:
    feed_cursor.execute(sql)
    data = feed_cursor.fetchall()
    with connection.cursor() as cursor:
        first_line = True
        insert_batch = []
        for index, line in enumerate(data):
            tweet = line['message']
            vader        = line['vader']
            sentiwordnet = line['sentiwordnet']
            sent    = vader
            if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 140:
                tweet = addslashes(tweet)
                sanitized_tweet = tokenize(tweet)
                sql = "INSERT INTO trinary_sentiment_dataset (id, sentiment, text, sanitized_text) VALUES ({id}, {sentiment}, '{text}', '{sanitized_tweet}')"
                sql = sql.format(id=index, sentiment=int(sent), text=tweet, sanitized_tweet = sanitized_tweet)
                insert_batch.append(sql)
                print(tweet[:140])
            if len(insert_batch) > 50000:
                for inst in insert_batch:
                    print(inst[:150])
                    cursor.execute(inst)
                connection.commit()
                insert_batch = []
        for inst in insert_batch:
            print(inst[:350])
            cursor.execute(inst)
        connection.commit()
        insert_batch = []
