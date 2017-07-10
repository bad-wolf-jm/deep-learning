# -*- coding: <utf8> -*-
import zipfile
import os
import html
import pymysql
import numpy as np
import pprint as pp
import math

LENGTH_CUTOFF = 10

connection = pymysql.connect(host='10.137.11.91',
                             user='jalbert',
                             password='gameloft2017',
                             db='tren_games',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

t_connection = pymysql.connect(host='localhost',
                               user='root',
                               password='root',
                               db='sentiment_analysis_data',
                               charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor)

tables = ['amazon__reviews',
          'facebook__comments',
          'glforums__posts',
          'googleplay__comments',
          'gyoutube__comments',
          'instagram__comments',
          'reddit__posts',
          'twitch__comments',
          'twitch__posts',
          'twitter__tweets',
          'windows__comments',
          'youtubeapi__comments']

sql_statements = []


def addslashes(s):
    if s == None:
        return ''
    d = {'"': '\\"', "'": "\\'", "\0": "\\\0", "\\": "\\\\"}
    return ''.join(d.get(c, c) for c in s)


total_size = 0
msg_count = 1

# if False:
with connection.cursor() as cursor:
    for table in tables:
        sql = "SELECT message FROM {table}".format(table=table)
        cursor.execute(sql)
        print(table)
        byte_lengths = []
        char_lengths = []
        byte_lengths_d = {}
        char_lengths_d = {}
        while True:
            data = cursor.fetchmany(10000)
            for row in data:
                row = bytes([x for x in row['message'].encode('utf8') if x != 0])
                if len(row) == 0:
                    continue
                try:
                    row = row.decode('utf8')
                    sql = "INSERT INTO byte2vec__training_strings (id, shuffle_id, text) VALUES ({msg_count}, {msg_count}, '{text}')"
                    sql = sql.format(msg_count=msg_count, text=addslashes(row))
                    sql_statements.append(sql)
                    total_size += len(row)
                    msg_count += 1
                except Exception as e:
                    print(e)
                    pass
            if not data:
                break
            with t_connection.cursor() as t_cursor:
                for i, sql in enumerate(sql_statements):
                    t_cursor.execute(sql)
                    print(i)
                sql_statements = []
                stats_sql = "INSERT INTO byte2vec__stats (id, num_bytes, num_messages) VALUES (0, {num_bytes}, {num_messages}) ON DUPLICATE KEY UPDATE num_bytes={num_bytes}, num_messages={num_messages}"
                stats_sql = stats_sql.format(num_bytes=total_size, num_messages=msg_count)
                t_cursor.execute(stats_sql)
            t_connection.commit()

with t_connection.cursor() as cursor:
    sql = """SELECT COUNT(id) as N FROM byte2vec__training_strings"""
    cursor.execute(sql)
    N = cursor.fetchone()['N']
    shuffle = np.random.permutation(N)
    for id_, perm_id in enumerate(shuffle):
        sql = """UPDATE byte2vec__training_strings SET shuffle_id={perm_id} WHERE id={id_}"""
        sql = sql.format(perm_id=perm_id, id_=id_)
        print(sql)
        cursor.execute(sql)
    t_connection.commit()


with t_connection.cursor() as cursor:
    sql = "SELECT text FROM byte2vec__training_strings"
    cursor.execute(sql)
    byte_frequencies = {i: 0 for i in range(256)}
    while True:
        data = cursor.fetchmany(1000)
        for row in data:
            message = row['text'].encode('utf8')
            for b in message:
                byte_frequencies[b] += 1
        if not data:
            break
    byte_insert_data = {}
    N = sum(byte_frequencies.values())
    sql = """INSERT INTO byte2vec__byte_frequencies (byte, frequency, probability, keep_probability, unigram_probability)
                VALUES ({byte}, {frequency}, {probability}, {keep_probability}, {unigram_probability})
                ON DUPLICATE KEY UPDATE frequency={frequency},
                                        probability={probability},
                                        keep_probability={keep_probability},
                                        unigram_probability={unigram_probability}"""
    Z = sum([math.pow(x, 0.75) for x in byte_frequencies.values()])
    for byte in byte_frequencies:
        occurence_probability = float(byte_frequencies[byte]) / N
        unigram_probability = math.pow(byte_frequencies[byte], 0.75) / Z
        keep_probability = 1
        if occurence_probability != 0:
            keep_probability = min((math.sqrt(occurence_probability / 0.001) + 1) * (0.001 / occurence_probability), 1)
        byte_insert_data[byte] = {'byte': byte,
                                  'frequency': byte_frequencies[byte],
                                  'probability': occurence_probability,
                                  'keep_probability': keep_probability,
                                  'unigram_probability': unigram_probability}
    for byte in byte_insert_data.values():
        x = sql.format(**byte)
        print (x)
        cursor.execute(x)
        t_connection.commit()
