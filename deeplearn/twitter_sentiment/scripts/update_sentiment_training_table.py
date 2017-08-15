import os
import sys
import pymysql
import numpy as np

ignore_bugs = '--ignore-bugs' in sys.argv

# connection to cms database
cms_connection = pymysql.connect(host='localhost', user='root', password='root',
                                 db='sentiment_analysis_data', charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

# connection to beta database, for user_input_sentiments
buzzometer_beta_connection = pymysql.connect(host='localhost',
                                             user='jalbert',
                                             port=6666,
                                             password='Blu3sdancing',
                                             db='tren_games',
                                             charset='utf8mb4',
                                             cursorclass=pymysql.cursors.DictCursor)

# connection to either dev database, or local database where the training table resides
buzzometer_dev_connection = pymysql.connect(host='localhost', user='root', password='root',
                                            db='sentiment_analysis_data', charset='utf8mb4',
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


# Source query for the cms database
cms_query = """ SELECT comment.message AS message, comments_flags.flag_id as sentiment
FROM (comment INNER JOIN comments_flags ON comment.id = comments_flags.comment_id)
WHERE comments_flags.flag_id between 1 and 5"""

cms_query_sentiment_map = {1:0, 2:3, 3:1, 4:2, 5:0}
if ignore_bugs:
    cms_query_sentiment_map[1]=None

# Source query for the user_assigned sentiments
tren_games_query = """SELECT message, user_assigned_sentiment as sentiment FROM {table} WHERE lang='en'
AND user_assigned_sentiment IS NOT NULL and user_assigned_sentiment != -3"""
tren_games_query_sentiment_map = {-1:0, 0:1, 1:2, -2:3}

dataset_insert_query = """INSERT INTO {table_name}
(id, sentiment, text, sanitized_text, message_length, sanitized_message_length)
VALUES ({id}, {sentiment}, '{message}', '{sanitized_message}', {message_length},
{sanitized_message_length})"""


def max_value(connection, table_name, field):
    with connection.cursor() as c:
        sql = 'select max({field}) as M from {table}'
        sql=sql.format(table=table_name,
                       field=field)
        c.execute(sql)
        return c.fetchone()['M']


def populate_table(table_name, source_query, source_connection, target_connection, sentiment_map):
    with source_connection.cursor() as source_cursor:
        with target_connection.cursor() as target_cursor:
            source_cursor.execute(source_query)
            entries = source_cursor.fetchall()
            I = max_value(target_connection, table_name, 'id')
            I = I+1 if I is not None else 0
            insert_batch = []
            for index, row in enumerate(entries):
                message = row['message']
                sentiment = sentiment_map[row['sentiment']]
                if sentiment is None:
                    continue
                print(index)
                message_length = len(message)
                sanitized_message = tokenize(message)
                sanitized_message_length = len(sanitized_message)
                sql = dataset_insert_query.format(id=I,
                                                  table_name=table_name,
                                                  sentiment=int(sentiment),
                                                  message=addslashes(message),
                                                  sanitized_message=addslashes(sanitized_message),
                                                  message_length=message_length,
                                                  sanitized_message_length=sanitized_message_length)
                insert_batch.append(sql)
                I += 1
                if len(insert_batch) > 50000:
                    for inst in insert_batch:
                        target_cursor.execute(inst)
                    target_connection.commit()
                    insert_batch = []
            for inst in insert_batch:
                target_cursor.execute(inst)
            target_connection.commit()

if ignore_bugs:
    training_table = 'neuronet_training_dataset'
else:
    training_table = 'neuronet_training_dataset_no_bug'

def fetch_cms_data(table_name):
    populate_table(table_name=table_name,
                   source_query=cms_query,
                   source_connection=cms_connection,
                   target_connection=buzzometer_dev_connection,
                   sentiment_map=cms_query_sentiment_map)

def fetch_buzzometer_data(table_name):
    tables = ['amazon__reviews',
              'facebook__comments',
              'glforums__posts',
              'googleplay__comments',
              'instagram__comments',
              'reddit__posts',
              'twitch__comments',
              'twitch__posts',
              'twitter__tweets',
              'windows__comments',
              'youtube__comments']

    for table in tables:
        populate_table(table_name=table_name,
                       source_query=tren_games_query.format(table=table),
                       source_connection=buzzometer_beta_connection,
                       target_connection=buzzometer_dev_connection,
                       sentiment_map=tren_games_query_sentiment_map)


def shuffle_table_rows(table_name, connection):
    with connection.cursor() as cursor:
        sql = """SELECT COUNT(id) as N FROM {table_name}""".format(table_name=table_name)
        cursor.execute(sql)
        N = cursor.fetchone()['N']
        shuffle = np.random.permutation(N)
        for id_, perm_id in enumerate(shuffle):
            sql = """UPDATE {table_name} SET shuffle_id={perm_id} WHERE id={id_}"""
            sql = sql.format(table_name=table_name, perm_id=perm_id, id_=id_)
            print(sql)
            cursor.execute(sql)
        connection.commit()


def truncate_table(table_name):
    with connection.cursor() as cursor:
        sql = """TRUNCATE {table_name}""".format(table_name=table_name)
        cursor.execute(sql)

truncate_table(table_name)
fetch_cms_data(training_table)
fetch_buzzometer_data(training_table)
shuffle_table_rows(training_table, buzzometer_dev_connection)
