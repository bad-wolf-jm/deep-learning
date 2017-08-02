#import zipfile
import pymysql
import pprint
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

def format_confusion_matrix(labels, true_labels, predicted_labels):
    matrix = {}
    for i in labels:
        for j in labels:
            matrix[i, j] = 0
    for t_l, p_l in zip(true_labels, predicted_labels):
        if (t_l, p_l) not in matrix:
                matrix[(t_l, p_l)] = 0
        matrix[(t_l, p_l)] += 1
    return [[i, j, matrix[j, i]] for i, j in matrix]

convert={-1:0, 0:1, 1:2, -2:3}

sql_base = """SELECT message, vader, sentiwordnet, user_assigned_sentiment FROM {table} WHERE lang='en' AND user_assigned_sentiment IS NOT NULL and user_assigned_sentiment != -3"""
index = 0
convert={-1:0, 0:1, 1:2, -2:3}
data_table=[]

with feed_connection.cursor() as feed_cursor:
    for table in tables:
        print(table)
        #sql = """SELECT message, vader, sentiwordnet FROM {table} WHERE lang='en'"""
        sql = sql_base.format(table=table)
        feed_cursor.execute(sql)
        data = feed_cursor.fetchall()
        data_table.extend(data)

vader_prediction = [convert[x['vader']] for x in data]
senti_prediction = [convert[x['sentiwordnet']] for x in data]
truth = [convert[x['user_assigned_sentiment']] for x in data]

x = [1 for (x, y) in zip(truth, vader_prediction) if x==y]

pprint.pprint(format_confusion_matrix([0,1,2,3], truth, vader_prediction))
pprint.pprint(format_confusion_matrix([0,1,2,3], truth, senti_prediction))
print(len(x), len(truth))
