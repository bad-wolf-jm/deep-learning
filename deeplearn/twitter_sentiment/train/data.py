from train.dbi import DBConnection

db_host = '127.0.0.1'
db_user = 'root'
db_password = ''

db_connection = DBConnection(host=db_host, user=db_user, password=db_password)
db_connection.connect('sentiment_analysis_data')


def count_rows(table, min_id=0, max_id=None):
    return db_connection.count_rows(table, 'id', min_id, max_id)


def _generate_batches(min_id=0, max_id=None, batch_size=10, epochs=None, sentiment_map=None, table=None):
    gen = db_connection.batches(table, 'id', ['sanitized_text', 'sentiment'], batch_size=batch_size, epochs=epochs)
    sentiment_map = sentiment_map or {}
    for b in iter(gen):
        batch_x = []
        batch_y = []
        for row in b:
            bytes_ = [ord(x) for x in row['sanitized_text'] if 0 < ord(x) < 256]
            batch_x.append(bytes_)
            batch_y.append([sentiment_map.get(row['sentiment'], row['sentiment'])])
        yield {'train_x': batch_x,
               'train_y': batch_y,
               'batch_number': gen.current_epoch_batch_number,
               'batches_per_epoch': gen.batches_per_epoch,
               'epoch_number': gen.current_epoch_number,
               'batch_index': gen.current_global_batch_number,
               'total_batches': gen.total_number_of_batches,
               'total_epochs': gen.number_of_epochs}


def generate_sentiment_batches(min_id=0, max_id=None, batch_size=10, epochs=None):
    for i in _generate_batches(min_id=min_id, max_id=max_id, batch_size=batch_size, epochs=epochs, sentiment_map={0: 0, 1: 1, -1: 2},
                               table='trinary_sentiment_dataset'):
        yield i


def sentiment_training_generator(batch_size=10, epochs=None, validation_size=None):
    if validation_size is not None:
        N = count_rows('trinary_sentiment_dataset')
        test = N // 100
        validation_iterator = generate_sentiment_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None)
    else:
        N = None
        test = 0
        validation_iterator = None
    batch_generator = generate_sentiment_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs)
    return {'train': batch_generator,
            'validation': validation_iterator}


def generate_cms_batches(min_id=0, max_id=None, batch_size=10, epochs=None):
    for i in _generate_batches(min_id=min_id, max_id=max_id, batch_size=batch_size, epochs=epochs,
                               table='cms_staging__dataset'):
        yield i


def cms_training_generator(batch_size=10, epochs=None, validation_size=None):
    if validation_size is not None:
        N = count_rows('trinary_sentiment_dataset')
        test = N // 100
        validation_iterator = generate_cms_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None)
    else:
        N = None
        test = 0
        validation_iterator = None
    batch_generator = generate_cms_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs)
    return {'train': batch_generator,
            'validation': validation_iterator}
