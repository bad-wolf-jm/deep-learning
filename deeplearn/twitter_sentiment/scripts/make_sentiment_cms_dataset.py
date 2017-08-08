#import zipfile
import pymysql
import numpy as np

LENGTH_CUTOFF = 10

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root',
                             db='sentiment_analysis_data',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

feed_connection = pymysql.connect(host='localhost',
                                  user='jalbert',
                                  port=6666,
                                  password='Blu3sdancing',
                                  db='tren_games',
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor)


tables = ['amazon__reviews',
          'facebook__comments',
          'glforums__posts',
          'googleplay__comments',
          #'gyoutube__comments',
          'instagram__comments',
          'reddit__posts',
          'twitch__comments',
          'twitch__posts',
          'twitter__tweets',
          'windows__comments',
          'youtube__comments']


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


sql_base = """SELECT message, user_assigned_sentiment FROM {table} WHERE lang='en' AND user_assigned_sentiment IS NOT NULL and user_assigned_sentiment != -3"""
index = 0
convert={-1:0, 0:1, 1:2, -2:3}

with feed_connection.cursor() as feed_cursor:
    for table in tables:
        sql = sql_base.format(table=table)
        feed_cursor.execute(sql)
        data = feed_cursor.fetchall()
        with connection.cursor() as cursor:
            first_line = True
            insert_batch = []
            for _, line in enumerate(data):
                tweet = line['message']
                #vader = line['vader']

                sentiment = convert[line['user_assigned_sentiment']]
                #sent = vader
                if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 512:
                    tweet_stats = {'char_length': len(tweet),
                                   'byte_length': len(tweet.encode('utf8'))}
                    sanitized_tweet = tokenize(tweet)
                    sanitized_tweet_stats = {'char_length': len(sanitized_tweet),
                                             'byte_length': len(sanitized_tweet.encode('utf8'))}

                    tweet = addslashes(tweet)
                    sanitized_tweet = addslashes(sanitized_tweet)
                    sql = """INSERT INTO user_cms_sentiment_dataset
                                            (id, sentiment, text, sanitized_text,
                                            char_length, byte_length,
                                            sanitized_char_length, sanitized_byte_length)
                            VALUES ({id}, {sentiment}, '{text}', '{sanitized_tweet}',
                                    {char_length}, {byte_length},
                                    {sanitized_char_length}, {sanitized_byte_length})"""
                    sql = sql.format(id=index,
                                     sentiment=int(sentiment),
                                     #sentiwordnet_sentiment=int(sentiwordnet),
                                     text=tweet,
                                     sanitized_tweet=sanitized_tweet,
                                     char_length=tweet_stats['char_length'],
                                     byte_length=tweet_stats['byte_length'],
                                     sanitized_char_length=sanitized_tweet_stats['char_length'],
                                     sanitized_byte_length=sanitized_tweet_stats['byte_length'])
                    insert_batch.append(sql)
                    index += 1
                    #print(tweet[:140])
                if len(insert_batch) > 50000:
                    for inst in insert_batch:
                        #print(inst[:150])
                        cursor.execute(inst)
                    connection.commit()
                    insert_batch = []
            for inst in insert_batch:
                #print(inst[:350])
                cursor.execute(inst)
            connection.commit()
            insert_batch = []


sql_base = """SELECT * FROM cms_staging__dataset"""
index += 1

convert={0:0, 1:3, 2:1, 3:2, 4:0}

with connection.cursor() as feed_cursor:
    feed_cursor.execute(sql_base)
    data = feed_cursor.fetchall()
    with connection.cursor() as cursor:
        first_line = True
        insert_batch = []
        for _, line in enumerate(data):
            line['id']=index
            line['sentiment'] = convert[line['sentiment']]
            line['text']=addslashes(line['text'])
            line['sanitized_text']=addslashes(line['sanitized_text'])
            sql = """INSERT INTO user_cms_sentiment_dataset
                                    (id, sentiment, text, sanitized_text,
                                    char_length, byte_length,
                                    sanitized_char_length, sanitized_byte_length)
                    VALUES ({id}, {sentiment}, '{text}', '{sanitized_text}',
                            {char_length}, {byte_length},
                            {sanitized_char_length}, {sanitized_byte_length})"""
            sql = sql.format(**line)
            insert_batch.append(sql)
            index += 1
            #print(line['text'][:140])
        if len(insert_batch) > 50000:
            for inst in insert_batch:
                #print(inst[:150])
                feed_cursor.execute(inst)
            connection.commit()
            insert_batch = []
    for inst in insert_batch:
        #print(inst[:350])
        feed_cursor.execute(inst)
    connection.commit()
    insert_batch = []






with connection.cursor() as cursor:
    sql = """SELECT COUNT(id) as N FROM user_cms_sentiment_dataset"""
    cursor.execute(sql)
    N = cursor.fetchone()['N']
    shuffle = np.random.permutation(N)
    for id_, perm_id in enumerate(shuffle):
        sql = """UPDATE user_cms_sentiment_dataset SET shuffle_id={perm_id} WHERE id={id_}"""
        sql = sql.format(perm_id=perm_id, id_=id_)
        print(sql)
        cursor.execute(sql)
    connection.commit()
