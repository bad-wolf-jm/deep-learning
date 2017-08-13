# -*- coding: <utf8> -*-


import zipfile
import os
import html
import pymysql
import numpy as np

LENGTH_CUTOFF = 10

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='sentiment_analysis_data',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

window_size = 4
byte_frequencies = {v:0 for v in range(256)}
joint_byte_frequencies = {}
for i in range(256):
    for j in range(256):
        joint_byte_frequencies[i,j]=0

batch_size = 1000
count_sql = "SELECT MAX(id) as max_id FROM twitter_binary_classification"
with connection.cursor() as cursor:
    cursor.execute(count_sql)
    max_id = cursor.fetchone()['max_id']
    num_batches = max_id // batch_size
    start_id = 0
    while start_id < max_id:
        fetch_sql = "SELECT id, text FROM twitter_binary_classification WHERE id BETWEEN {start_id} AND {end_id}"
        fetch_sql = fetch_sql.format(start_id = start_id, end_id = start_id + batch_size)
        cursor.execute(fetch_sql)
        rows = cursor.fetchall()
        if rows:
            new_max_id = max([x['id'] for x in rows])
            for row in rows:
                if row['text'] is not None:
                    text = [x for x in bytes(row['text'].encode('utf-8'))]
                    L = len(text)
                    text = [0]*window_size + text + [0]*window_size
                    start = window_size
                    for index in range(start, start+L, 1):
                        char = text[index]
                        byte_frequencies[char] += 1
                        for context_char in range(-window_size, window_size + 1):
                            if context_char != 0:
                                context = text[index+context_char]
                                joint_byte_frequencies[char, context] += 1
                    print (row['id'])#, text[:20])
            start_id = new_max_id
        else:
            break

    total_frequency = sum(byte_frequencies.values())
    for b in byte_frequencies:
        sql = """INSERT INTO byte_distribution (byte, frequency, probability)
                 VALUES ({byte}, {frequency}, {probability})"""
        frequency = byte_frequencies[b]
        probability = float(frequency) / total_frequency
        byte = b
        sql = sql.format(byte=byte, frequency=frequency, probability=probability)
        cursor.execute(sql)
        #print(b, joint_byte_frequencies[b])
        print(b, byte_frequencies[b])


    total_frequency = sum(joint_byte_frequencies.values())
    for b in joint_byte_frequencies:
        sql = """INSERT INTO joint_byte_distribution (query_byte, context_byte, frequency, probability)
                 VALUES ({query_byte}, {context_byte}, {frequency}, {probability})"""
        frequency = joint_byte_frequencies[b]
        probability = float(frequency) / total_frequency
        query_byte, context_byte = b
        sql = sql.format(query_byte=query_byte, context_byte=context_byte, frequency=frequency, probability=probability)
        cursor.execute(sql)
        print(b, joint_byte_frequencies[b])
    connection.commit()
