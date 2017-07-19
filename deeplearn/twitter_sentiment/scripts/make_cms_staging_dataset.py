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

query = """SELECT cms_staging__comment.message AS text,
                  cms_staging__comments_flags.flag_id AS flag_id,
                  cms_staging__flag.name AS flag_name,
                  cms_staging__flag.tab as tab
           FROM (cms_staging__comment INNER JOIN cms_staging__comments_flags ON
                    cms_staging__comment.id = cms_staging__comments_flags.comment_id)
                    INNER JOIN cms_staging__flag ON
                        cms_staging__flag.id = cms_staging__comments_flags.flag_id
           WHERE cms_staging__comments_flags.flag_id between 1 and 5"""

with connection.cursor() as cursor:
    cursor.execute(query)
    entries = cursor.fetchall()
    for row in entries:
        print(row)

sys.exit(0)

for i, g in enumerate(glob.glob(_file)):
    file_name = os.path.basename(g)
    id_, gender, age, industry, sign, _ = file_name.split('.')

    print (i, g)

    try:
        str_ = open(g).read().encode('utf8')
        bl = etree.fromstring(str_, parser=parser)
        for entry in bl.iter('post'):
            sql = """INSERT INTO blog_corpus (id, blogger_id, gender, age, char_length, byte_length, text) VALUES ({id}, {blogger_id}, '{gender}', {age}, {char_length}, {byte_length}, '{text}')"""
            sql=sql.format(id=key,
                           blogger_id=id_,
                           gender=gender,
                           age=age,
                           text=addslashes(entry.text),
                           char_length=len(entry.text),
                           byte_length=len(entry.text.encode('utf8')))
            sql_insert_statements.append(sql)
            key += 1
            #print(entry.text.strip('\n'))
    except Exception as e:
        XX += 1
        print('------------------', e)

print (XX)
with connection.cursor() as cursor:
    for i, stat in enumerate(sql_insert_statements):
        print(stat[:350].replace('\n', ' '))
        cursor.execute(stat)
        print (i, len(sql_insert_statements))
connection.commit()
