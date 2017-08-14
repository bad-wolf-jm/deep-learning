from train.dbi import DBConnection

db_host = '127.0.0.1'
db_user = 'root'
db_password = 'root'

db_connection = DBConnection(host=db_host, user=db_user, password=db_password)
db_connection.connect('sentiment_analysis_data')


class DatabaseSource(object):
    def __init__(self, field_name=None, text_column='sanitized_text', min_id=0, max_id=None, batch_size=10, epochs=None, sentiment_map=None, table=None):
        self.field_name = field_name
        self.text_column = text_column,
        self.min_id = min_id
        self.max_id = max_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.table = table

    def count_rows(table, min_id=0, max_id=None):
        return db_connection.count_rows(table, 'id', min_id, max_id)

    def _generate_batches(field_name=None, text_column='sanitized_text', min_id=0, max_id=None, batch_size=10, epochs=None, sentiment_map=None, table=None):
        gen = db_connection.batches(table, 'shuffle_id', [text_column, field_name], batch_size=batch_size, epochs=epochs)
        sentiment_map = sentiment_map or {}
        for b in iter():
            batch_x = []
            batch_y = []
            for row in b:
                batch_x.append(row[text_column])
                batch_y.append(row[field_name])
            yield {'train_x': batch_x,
                   'train_y': batch_y,
                   'batch_number': gen.current_epoch_batch_number,
                   'batches_per_epoch': gen.batches_per_epoch,
                   'epoch_number': gen.current_epoch_number,
                   'batch_index': gen.current_global_batch_number,
                   'total_batches': gen.total_number_of_batches,
                   'total_epochs': gen.number_of_epochs}


class CMSUserInputDataset(DatabaseSource):
    display_name = "User Input and CMS Flagging Dataset"
    type = 'categorical_data'
    language = 'en'
    number_of_classes = 4
    category_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Irrelevant'}
