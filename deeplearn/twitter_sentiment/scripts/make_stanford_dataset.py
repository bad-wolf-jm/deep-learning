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


data_file = os.path.expanduser('datasets/stanford-sentiment-treebank/dictionary.txt')
sentiment_data_file = os.path.expanduser('datasets/stanford-sentiment-treebank/sentiment_labels.txt')

print (os.getcwd())


d_file = open(data_file)
s_file = open(sentiment_data_file)
#parser = etree.XMLParser(recover=True)
#XX = 0
def addslashes(s):
    if s == None:
        return ''
    d = {'"': '\\"', "'": "\\'", "\0": "\\\0", "\\": "\\\\"}
    return ''.join(d.get(c, c) for c in s)

sql_insert_statements = []
key=0

phrases = {}
sentiments = {}

for line in d_file.readlines():
    phrase, phrase_id = line.split('|')
    phrases[int(phrase_id[:-1])] = phrase
s_file.readline()
for line in s_file.readlines():
    phrase_id, sentiment = line.split('|')
    sentiments[int(phrase_id)] = float(sentiment)

ids = sorted(phrases.keys())

def split(x, n):
    for i in range(n):
        if x > float(i) / n and  x <= float(i+1) / n:
            return i
    return n

with connection.cursor() as cursor:
    for id_ in ids:
        print(id_, phrases[id_][:30] + "."*max(30-len(phrases[id_]), 0), sentiments.get(id_, None))
        sql = """INSERT INTO sst_phrase_dataset (id, char_length, byte_length, text, sentiment_value, sentiment_three, sentiment_five)
                    VALUES ({id}, {char_length}, {byte_length}, '{text}', {sentiment_value},{sentiment_three}, {sentiment_five})"""
        sql=sql.format(id=id_,
                       sentiment_value=sentiments[id_],
                       sentiment_three=split(sentiments[id_], 3),
                       sentiment_five=split(sentiments[id_], 5),
                       text=addslashes(phrases[id_]),
                       char_length=len(phrases[id_]),
                       byte_length=len(phrases[id_].encode('utf8')))
        sql_insert_statements.append(sql)

    for inst in sql_insert_statements:
        print(inst[:350])
        cursor.execute(inst)
    connection.commit()


with connection.cursor() as cursor:
    sql = """SELECT COUNT(id) as N FROM sst_phrase_dataset"""
    cursor.execute(sql)
    N = cursor.fetchone()['N']
    shuffle = np.random.permutation(N)
    for id_, perm_id in enumerate(shuffle):
        sql = """UPDATE sst_phrase_dataset SET shuffle_id={perm_id} WHERE id={id_}"""
        sql = sql.format(perm_id=perm_id, id_=id_)
        print(sql)
        cursor.execute(sql)
    connection.commit()
