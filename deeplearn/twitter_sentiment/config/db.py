db_name = 'sentiment_analysis_data'
db_host = '127.0.0.1'
db_user = 'root'
db_password = ''
db_charset = 'utf8mb4'
db_shuffle_rows = False


def fill_arg_parser(parser):
    parser.add_argument('-d', '--database', dest='database',
                        type=str,
                        default=db_name,
                        help='The training database name')
    parser.add_argument('-H', '--host', dest='host',
                        type=str,
                        default=db_host,
                        help='The training database IP')
    parser.add_argument('-u', '--user', dest='user',
                        type=str,
                        default=db_user,
                        help='The training database username')
    parser.add_argument('-p', '--password', dest='password',
                        type=str,
                        default=db_password,
                        help='The training database password')
    parser.add_argument('-c', '--charset', dest='charset',
                        type=str,
                        default=db_charset,
                        help='The training database password')
    parser.add_argument('-S', '--shuffle-rows', dest='shuffle_rows',
                        type=bool,
                        default=db_shuffle_rows,
                        help='If this option is present, the rows of the database will be shuffled')
