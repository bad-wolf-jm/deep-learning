from train.dbi import DBConnection

db_host = '127.0.0.1'
db_user = 'root'
db_password = 'root'

db_connection = DBConnection(host=db_host, user=db_user, password=db_password)
db_connection.connect('sentiment_analysis_data')


def count_rows(table, min_id=0, max_id=None):
    return db_connection.count_rows(table, 'id', min_id, max_id)


def _generate_batches(field_name=None, text_column='sanitized_text', min_id=0, max_id=None, batch_size=10, epochs=None, sentiment_map=None, table=None):
    gen = db_connection.batches(table, 'shuffle_id', [text_column, field_name], batch_size=batch_size, epochs=epochs)
    sentiment_map = sentiment_map or {}
    for b in iter(gen):
        batch_x = []
        batch_y = []
        for row in b:
            bytes_ = [ord(x) for x in row[text_column] if 0 < ord(x) < 256]
            batch_x.append(bytes_)
            batch_y.append([sentiment_map.get(row[field_name], row[field_name])])
        #print([row['shuffle_id'] for row in b]) 
        yield {'train_x': batch_x,
               'train_y': batch_y,
               'batch_number': gen.current_epoch_batch_number,
               'batches_per_epoch': gen.batches_per_epoch,
               'epoch_number': gen.current_epoch_number,
               'batch_index': gen.current_global_batch_number,
               'total_batches': gen.total_number_of_batches,
               'total_epochs': gen.number_of_epochs}


def generate_sentiment_batches(min_id=0, max_id=None, batch_size=10, epochs=None, field_name=None):
    for i in _generate_batches(min_id=min_id, max_id=max_id, batch_size=batch_size, epochs=epochs, sentiment_map={0: 0, 1: 1, -1: 2},
                               table='trinary_sentiment_dataset', field_name=field_name):
        yield i


def vader_sentiment_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('trinary_sentiment_dataset')
        test = N // 100
        validation_iterator = generate_sentiment_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None, field_name='vader_sentiment')
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('trinary_sentiment_dataset')
        test = N // 100
        test_iterator = generate_sentiment_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None, field_name='vader_sentiment')
    else:
        N = None
        test = 0
        test_iterator = None

    batch_generator = generate_sentiment_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs, field_name='vader_sentiment')
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}


def sentiwordnet_sentiment_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('trinary_sentiment_dataset')
        test = N // 100
        validation_iterator = generate_sentiment_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None, field_name='sentiwordnet_sentiment')
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('trinary_sentiment_dataset')
        test = N // 100
        test_iterator = generate_sentiment_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None, field_name='sentiwordnet_sentiment')
    else:
        N = None
        test = 0
        test_iterator = None

    batch_generator = generate_sentiment_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs, field_name='sentiwordnet_sentiment')
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}


def generate_cms_batches(min_id=0, max_id=None, batch_size=10, epochs=None):
    for i in _generate_batches(min_id=min_id, max_id=max_id, batch_size=batch_size, epochs=epochs,
                               table='cms_staging__dataset', field_name='sentiment'):
        yield i


def cms_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('cms_staging__dataset')
        test = N // 100
        validation_iterator = generate_cms_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None)
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('cms_staging__dataset')
        test = N // 100
        test_iterator = generate_cms_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None)
    else:
        N = None
        test = 0
        test_iterator = None
    batch_generator = generate_cms_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs)
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}



def generate_user_cms_batches(min_id=0, max_id=None, batch_size=10, epochs=None):
    for i in _generate_batches(min_id=min_id, max_id=max_id, batch_size=batch_size, epochs=epochs,
                               table='user_cms_sentiment_dataset', field_name='sentiment'):
        yield i


def user_cms_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('user_cms_sentiment_dataset')
        test = N // 50
        validation_iterator = generate_user_cms_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None)
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('user_cms_sentiment_dataset')
        test = N // 50
        test_iterator = generate_user_cms_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None)
    else:
        N = None
        test = 0
        test_iterator = None
    batch_generator = generate_user_cms_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs)
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}








def generate_sstb_batches(min_id=0, max_id=None, batch_size=10, epochs=None, field_name=None):
    for i in _generate_batches(min_id=min_id, max_id=max_id, batch_size=batch_size, epochs=epochs,
                               table='sst_phrase_dataset', field_name=field_name, text_column='text'):
        yield i


def sstb3_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        validation_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None, field_name='sentiment_three')
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        test_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None, field_name='sentiment_three')
    else:
        N = None
        test = 0
        test_iterator = None
    batch_generator = generate_sstb_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs, field_name='sentiment_three')
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}


def sstb5_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        validation_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None, field_name='sentiment_five')
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        test_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None, field_name='sentiment_five')
    else:
        N = None
        test = 0
        test_iterator = None
    batch_generator = generate_sstb_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs, field_name='sentiment_five')
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}


#foo = sstb5_training_generator(batch_size=150, epochs=2, validation_size=200, test_size=300)
# for batch in foo['train']:
#    print(batch['total_batches'], batch['batch_index'])
# sys.exit(0)


def generate_blog_batches(min_id=0, max_id=None, batch_size=10, epochs=None, field_name=None):
    for i in _generate_batches(min_id=min_id, max_id=max_id, batch_size=batch_size, epochs=epochs,
                               table='cms_staging__dataset', field_name=field_name):
        yield i


def blog_gender_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        validation_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None, field_name='sentiment_three')
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        test_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None, field_name='sentiment_three')
    else:
        N = None
        test = 0
        test_iterator = None
    batch_generator = generate_sstb_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs, field_name='sentiment_three')
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}


def blog_age_training_generator(batch_size=10, epochs=None, validation_size=None, test_size=None):
    if validation_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        validation_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=validation_size, epochs=None, field_name='sentiment_five')
    else:
        N = None
        test = 0
        validation_iterator = None
    if test_size is not None:
        N = count_rows('sst_phrase_dataset')
        test = N // 100
        test_iterator = generate_sstb_batches(min_id=0, max_id=test, batch_size=test_size, epochs=None, field_name='sentiment_five')
    else:
        N = None
        test = 0
        test_iterator = None
    batch_generator = generate_sstb_batches(min_id=test + 1, batch_size=batch_size, epochs=epochs, field_name='sentiment_five')
    return {'train': batch_generator,
            'validation': validation_iterator,
            'test': test_iterator}
