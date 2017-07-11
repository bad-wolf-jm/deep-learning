import time
from train.summary import TrainingSummary
from stream.sender import DataStreamer
from stream.receiver import DataReceiver


class TrainingDataStreamer(object):
    def __init__(self, validation_interval=None, summary_span=None):
        super(TrainingDataStreamer).__init__()
        self.validation_interval = validation_interval
        self.batch_index = 0
        self._epoch_number = None
        self._total_epochs = None
        self._batch_number = None
        self._batches_per_epoch = None
        self._batch_index = None
        self._total_batches = None
        self.summary = TrainingSummary(summary_span=summary_span,
                                       fields=['loss',
                                               'accuracy',
                                               'batch_time',
                                               'training_time',
                                               'travel_time'])

    def __default_validation_iterator(self):
        while True:
            yield None

    def send_to_streamer(self, streamer, action, data, summary):
        t_0 = time.time()
        vals = streamer.send({'action': action,
                              'payload': {'train_x': data['train_x'],
                                          'train_y': data['train_y']}})
        batch_total_time = time.time() - t_0
        vals = vals['return']
        y = vals['time']
        travel_time = batch_total_time - y
        self.summary.add_to_summary(summary,
                                    self.batch_index,
                                    loss=vals['loss'],
                                    accuracy=vals['accuracy'],
                                    batch_time=batch_total_time,
                                    training_time=vals['time'],
                                    travel_time=travel_time)

    def get_train_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.summary.get_summary('train',
                                     min_batch_index=min_batch_index,
                                     max_batch_index=max_batch_index)
        x['epoch_number'] = self._epoch_number
        x['total_epochs'] = self._total_epochs
        x['batch_number'] = self._batch_number
        x['batches_per_epoch'] = self._batches_per_epoch
        x['batch_index'] = self._batch_index
        x['total_batches'] = self._total_batches
        return x

    def get_validation_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.summary.get_summary('validation',
                                     min_batch_index=min_batch_index,
                                     max_batch_index=max_batch_index)
        return x

    def _update_progress_info(self, training_batch):
        self._epoch_number = training_batch.get('epoch_number', None)
        self._total_epochs = training_batch.get('total_epochs', None)
        self._batch_number = training_batch.get('batch_number', None)
        self._batches_per_epoch = training_batch.get('batches_per_epoch', None)
        self._batch_index = training_batch.get('batch_index', None)
        self._total_batches = training_batch.get('total_batches', None)

    def stream(self, training_data_generator, validation_data_generator=None, host=None, port=None):
        streamer = DataStreamer(host=host, post=port)
        bar = DataReceiver(port=9977)
        bar.register_action_handler('get_train_batch_data', self.get_train_summary)
        bar.register_action_handler('get_validation_batch_data', self.get_validation_summary)
        bar.start(True)
        validation_iterator = validation_data_generator or self.__default_validation_iterator()
        for training_batch in training_data_generator:
            self._update_progress_info(training_batch)
            self.send_to_streamer(streamer, 'train', training_batch, 'train')
            self.batch_index += 1
            if training_batch['batch_index'] % self.validation_interval == 0:
                validation_batch = next(validation_iterator)
                if validation_batch is not None:
                    self.send_to_streamer(streamer, 'validate', validation_batch, 'validation')
