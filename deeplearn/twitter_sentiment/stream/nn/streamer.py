import time
from train.summary import TrainingSummary
from stream.sender import DataStreamer

class TrainingDataStreamer(object):
    def __init__(self, validation_interval=None, summary_span=None):
        super(TrainingDataStream).__init__()
        self.validation_interval = validation_interval
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
                                    loss=vals['loss'],
                                    accuracy=vals['accuracy'],
                                    batch_time=batch_total_time,
                                    training_time=vals['time'],
                                    travel_time=travel_time)

    def stream(self, training_data_generator, validation_data_generator=None, host=None, port=None):
        streamer = DataStreamer(host=host, post=port)
        validation_iterator = validation_data_generator or self.__default_validation_iterator()
        for training_batch in training_data_generator:
            self.send_to_streamer(streamer, 'train', training_batch, 'train')
            if training_batch['batch_index'] % self.validation_interval == 0:
                validation_batch = next(validation_iterator)
                if validation_batch is not None:
                    self.send_to_streamer(streamer, 'validate', validation_batch, 'validation')
