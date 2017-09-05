import os
import sys
import pymysql
import numpy as np

ignore_bugs = '--ignore-bugs' in sys.argv

# connection to cms database
cms_connection = pymysql.connect(host='localhost', user='root', password='root',
                                 db='sentiment_analysis_data', charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)



sql = """SELECT sanitized_text FROM neuronet_training_dataset"""

chrs = {}
with cms_connection.cursor() as cursor:
    cursor.execute(sql)
    data = cursor.fetchall()
    for t in data:
        text = t['sanitized_text']
        for ch in text:
            if ch not in chrs:
                chrs[ch] = 0
            chrs[ch] += 1
    print(len(chrs))
    print (chrs)
    chs = sorted(chrs.keys(), key=lambda x:chrs[x])
    for i, x in enumerate(chs):
        print(i, x, '   ;', x.encode('utf8'), ord(x), chrs[x])
