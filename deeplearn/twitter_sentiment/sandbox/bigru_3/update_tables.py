import os
import tensorflow as tf
from bidirectional_gru import Tweet2Vec_BiGRU
import pymysql
import time


graph=tf.Graph()
session=tf.Session(graph=graph)
model = Tweet2Vec_BiGRU()

def initialize():  # , session, training=True, resume=False):
    weight_root = os.path.join(os.path.dirname(__file__), '.weights')
    #model = Tweet2Vec_BiGRU()
    with graph.as_default():
        model.build_inference_model()
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(weight_root, "model.ckpt"))
        print('model restored')

def compute_batch( batch_input):
    batch_x = [model.pad([ord(x) for x in element], 256) for element in batch_input]
    vals = session.run([tf.cast(tf.argmax(model.graph_output, 1), tf.uint8)],
                feed_dict={model.input: batch_x})
    #print(vals)
    return vals[0]


#connection = pymysql.connect(host='localhost',
#                             user='root',
#                             password='root',
#                             db='tren_games',
#                             charset='utf8mb4',
#                             cursorclass=pymysql.cursors.DictCursor)

connection = pymysql.connect(host='10.137.11.91',
                             user='jalbert',
                             password='gameloft2017',
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

def main():
    initialize()
    with connection.cursor() as cursor:
        for table in tables:
            print(table)
            #x = """SELECT COUNT(id) as N FROM {table} WHERE lang='en'"""
            #x = x.format(table=table)
            #cursor.execute(x)
            #N = cursor.fetchone()['N']



            sql = """SELECT id, message FROM {table} WHERE lang='en' and neuronet_sentiment IS NULL"""
            update_sql = "UPDATE {table} SET neuronet_sentiment={sentiment} WHERE id={id}"
            sql = sql.format(table=table)
            cursor.execute(sql)
            data = cursor.fetchall()
            sentiment_convert = {0:-1, 1:0, 2:1, 3:-2}
            while len(data) > 0:
                t_0 = time.time()
                batch = data[:5000]
                _ = [tokenize(x['message']) for x in batch]
                message_batch = _
                ids = [x['id'] for x in batch]
                batch_sentiments = compute_batch(message_batch)
                for id, sentiment in zip(ids, batch_sentiments):
                    if isinstance(id, str):
                        id = "'{id}'".format(id=id)
                    s = update_sql.format(table=table, id=id, sentiment=sentiment_convert[sentiment])
                    cursor.execute(s)
                connection.commit()
                t = time.time() - t_0
                data=data[5000:]
                print('{t:.2f}'.format(t=t), 'REMAINING: ', len(data))


if __name__ == '__main__':
    main()
