import pymysql


class DBIterator(object):
    def __init__(self, batch_generator):
        super(DBConnection, self).__init__(self)
        self._batch_generator = batch_generator
        self.columns = None
        self.batches_per_epoch = None
        self.total_number_of_batches = None
        self.number_of_epochs = None
        self.current_epoch_batch_number = None
        self.current_global_batch_number = None
        self.current_epoch_number = None

    def __iter__(self):
        for item in self._batch_generator:
            self.current_epoch_number = item['epoch']
            self.current_epoch_batch_number = item['batch_number']
            self.current_global_batch_number = item['batch_index']
            yield item['batch']


class DBConnection(object):
    def __init__(self, host='localhost', user='root', password='root'):
        super(DBConnection, self).__init__(self)
        self._host = host
        self._username = user
        self._password = password
        self._connection = None
        self._database_name = None

    def connect(self, database_name):
        self._database_name = database_name
        self._connection = pymysql.connect(host=self._host,
                                           user=self._user,
                                           password=self._password,
                                           db=self._database_name,
                                           charset='utf8mb4',
                                           cursorclass=pymysql.cursors.DictCursor)

    def count_rows(self, table_name, index_column, min_id=None, max_id=None):
        with self._connection.cursor() as cursor:
            if min_id is None and max_id is None:
                c = "SELECT COUNT({index_column}) as N from {table_name}"
                c = c.format(table_name=table_name, index_column=index_column)
            elif min_id is None and max_id is not None:
                c = "SELECT COUNT({index_column}) as N from {table_name} WHERE shuffle_id < {max_id}"
                c = c.format(table_name=table_name, index_column=index_column, max_id=max_id)
            elif min_id is not None and max_id is None:
                c = "SELECT COUNT({index_column}) as N from {table_name} WHERE shuffle_id >= {min_id}"
                c = c.format(table_name=table_name, index_column=index_column, min_id=min_id)
            else:
                c = "SELECT COUNT({index_column}) as N from {table_name} WHERE shuffle_id BETWEEN {min_id} AND {max_id}"
                c = c.format(table_name=table_name, index_column=index_column, max_id=max_id, min_id=min_id)
            cursor.execute(c)
            N = cursor.fetchone()['N']
            return N

    def _get_batch(self, table_name, index_column, select_columns=None, batch_size=100, starting_id=0, record_count=None):
        with self._connection.cursor() as cursor:
            data = []
            remaining = batch_size
            while remaining > 0:
                select_columns = ', '.join(select_columns) or "*"
                sql = "SELECT {select_columns} FROM {table} WHERE {index_column} BETWEEN {start_id} AND {end_id}"
                sql = sql.format(select_columns=select_columns,
                                 table_name=table_name,
                                 index_column=index_column,
                                 start_id=starting_id,
                                 end_id=starting_id + remaining)
                cursor.execute(sql)
                query_data = cursor.fetchall()
                if len(query_data) == 0:
                    starting_id += remaining
                    starting_id %= record_count
                max_id = max([x[index_column] for x in query_data])
                data.extend(query_data)
                starting_id = max_id
                remaining -= len(query_data)

            batch = data[:batch_size]
            max_id = max([x[index_column] for x in data])
            return [max_id, batch]

    def _generate_all_batches(self, table_name, index_column, select_columns=None,  min_id=None, max_id=None, batch_size=10, epochs=None):
        I = 0
        epoch = 1
        while (epochs is None) or (epoch > epochs):
            offset = min_id
            for batch in range(batches_per_epoch):
                o, batch = self._get_batch(cursor, starting_id=offset, batch_size=batch_size, record_count=N)
                I += 1
                yield {'batch': batch,
                       'batch_number': batch,
                       'epoch_number': epoch,
                       'batch_index': I}
                offset = o
            epoch += 1

    def batches(self, table_name, index_column, select_columns=None,  min_id=None, max_id=None, batch_size=10, epochs=None):
        N = self.count_rows(table_name, index_column, min_id=min_id, max_id=max_id)
        max_id = max_id or N
        total = None
        total_num_batches = None
        if epochs is not None:
            total = N * epochs
            total_num_batches = total // batch_size
        batches_per_epoch = N // batch_size
        I = 0
        epoch = 1
        generator = self._generate_all_batches(table_name, index_column, select_columns=select_columns,  min_id=min_id, max_id=max_id, batch_size, epochs)
        iterator = DBIterator(generator)
        iterator.columns = select_columns
        iterator.batches_per_epoch = batches_per_epoch
        iterator.total_number_of_batches = total_num_batches
        iterator.number_of_epochs = epochs
        return iterator
