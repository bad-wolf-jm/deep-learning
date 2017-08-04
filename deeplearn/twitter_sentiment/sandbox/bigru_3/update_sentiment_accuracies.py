import pymysql
connection = pymysql.connect(host='localhost',
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


def compute_accuracy(table, algorithm):
    count_sql = "SELECT COUNT(id) as N FROM {table} WHERE lang='en' AND user_assigned_sentiment IS NOT NULL;"
    count_match_sql = "SELECT COUNT(id) as N FROM {table} WHERE lang='en' AND {algorithm}=user_assigned_sentiment AND user_assigned_sentiment IS NOT NULL;"
    count_sql = count_sql.format(table=table)
    count_match_sql = count_match_sql.format(table=table, algorithm=algorithm)
    with connection.cursor() as c:
        c.execute(count_sql)
        N = c.fetchone()['N']
        c.execute(count_match_sql)
        A = c.fetchone()['N']
        print(table, algorithm, A, N)
        return float(A) / N if N != 0 else 0



for table_name in tables:
    vader_acc = compute_accuracy(table_name, 'vader')
    senti_acc = compute_accuracy(table_name, 'sentiwordnet')
    neuro_acc = compute_accuracy(table_name, 'neuronet_sentiment')
    print(table_name, vader_acc, senti_acc, neuro_acc)
