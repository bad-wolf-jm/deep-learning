#import zipfile
import os
import pymysql
import numpy as np
import glob
#import elementtr
from lxml import etree
import xml.etree.ElementTree as ET

LENGTH_CUTOFF = 10

### http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='root',
                             db='sentiment_analysis_data',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


_file = os.path.expanduser('~/Desktop/blogs/*.xml')
parser = etree.XMLParser(recover=True)
XX = 0
def addslashes(s):
    if s == None:
        return ''
    d = {'"': '\\"', "'": "\\'", "\0": "\\\0", "\\": "\\\\"}
    return ''.join(d.get(c, c) for c in s)

sql_insert_statements = []
key=0

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



query = """SELECT cms_staging__comment.message AS text,
                  cms_staging__comments_flags.flag_id AS flag_id,
                  cms_staging__flag.name AS flag_name,
                  cms_staging__flag.tab as tab
           FROM (cms_staging__comment INNER JOIN cms_staging__comments_flags ON
                    cms_staging__comment.id = cms_staging__comments_flags.comment_id)
                    INNER JOIN cms_staging__flag ON
                        cms_staging__flag.id = cms_staging__comments_flags.flag_id
           WHERE cms_staging__comments_flags.flag_id between 1 and 5"""
insert_batch = []
with connection.cursor() as cursor:
    cursor.execute(query)
    entries = cursor.fetchall()
    I = 0
    for index, row in enumerate(entries):
        tweet = row['text']
        sent = row['flag_id']
        if len(tweet) >= LENGTH_CUTOFF and len(tweet) <= 1024:
            tweet_stats = {'char_length': len(tweet),
                           'byte_length': len(tweet.encode('utf8'))}
            sanitized_tweet = tokenize(tweet)
            sanitized_tweet_stats = {'char_length': len(sanitized_tweet),
                                     'byte_length': len(sanitized_tweet.encode('utf8'))}

            tweet = addslashes(tweet)
            sanitized_tweet = addslashes(sanitized_tweet)
            sql = """INSERT INTO cms_staging__dataset
                                    (id, sentiment, text, sanitized_text,
                                    char_length, byte_length,
                                    sanitized_char_length, sanitized_byte_length)
                    VALUES ({id}, {sentiment}, '{text}', '{sanitized_tweet}',
                            {char_length}, {byte_length},
                            {sanitized_char_length}, {sanitized_byte_length})"""
            sql = sql.format(id=I,
                             sentiment=int(sent),
                             text=tweet,
                             sanitized_tweet=sanitized_tweet,
                             char_length=tweet_stats['char_length'],
                             byte_length=tweet_stats['byte_length'],
                             sanitized_char_length=sanitized_tweet_stats['char_length'],
                             sanitized_byte_length=sanitized_tweet_stats['byte_length'])
            insert_batch.append(sql)
            I += 1
            print(tweet[:256])
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


with connection.cursor() as cursor:
    sql = """SELECT COUNT(id) as N FROM cms_staging__dataset"""
    cursor.execute(sql)
    N = cursor.fetchone()['N']
    shuffle = np.random.permutation(N)
    for id_, perm_id in enumerate(shuffle):
        sql = """UPDATE cms_staging__dataset SET shuffle_id={perm_id} WHERE id={id_}"""
        sql = sql.format(perm_id=perm_id, id_=id_)
        print(sql)
        cursor.execute(sql)
    connection.commit()
