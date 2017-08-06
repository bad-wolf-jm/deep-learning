import time
import numpy as np
import tensorflow as tf
from train.summary import StreamSummary
import datetime
from train.datasources import get_dataset_specs


class InfiniteLoss(Exception):
    pass

class BatchData(object):
    def __init__(self, loss=0.0, accuracy=0.0, output=None):
        super(BatchData, self).__init__()
        self.loss = loss
        self.accuracy = accuracy
        self.output = output

class TrainData(BatchData):
    pass

class ValidationData(BatchData):
    pass

class TestData(BatchData):
    pass


class TrainingSupervisor(object):
    def __init__(self, session = None, model=None,
                 validation_interval=None, test_interval=None,
                 summary_span=None):
        super(TrainingSupervisor).__init__()
        self.model = model
        self.batch_index = 0
        self.validation_interval = validation_interval
        self.test_interval = test_interval  # test_interval in seconds

        fields = ['loss', 'accuracy']
        self.train_summary = StreamSummary(summary_span, fields)
        self.validation_summary = StreamSummary(summary_span, fields)
        self.training_time_summary = StreamSummary(summary_span, ['time'])
        self.batch_index = 0
        self._epoch_number = None
        self._total_epochs = None
        self._batch_number = None
        self._batches_per_epoch = None
        self._batch_index = None
        self._total_batches = None
        self._session = session
        self.data = get_dataset_specs(self.model.data)


    @property
    def batch_number(self):
        return self._batch_number

    @property
    def global_batch_index(self):
        return self._total_batches

    @property
    def batches_per_epoch(self):
        return self._batches_per_epoch

    @property
    def epoch_number(self):
        return self._epoch_number

    @property
    def number_of_epochs(self):
        return self._total_epochs

    @property
    def batch_time(self):
        x = self.training_time_summary.stats(fields=['time'], backlog=20)
        x = x['time']['mean']
        return datetime.timedelta(seconds=x)

    @property
    def epoch_time(self):
        x = self.training_time_summary.stats(fields=['time'], backlog=20)
        x = x['time']['mean']
        return datetime.timedelta(seconds=self.batches_per_epoch * x)

    @property
    def epoch_elapsed_time(self):
        x = self.training_time_summary.stats(fields=['time'], backlog=20)
        x = x['time']['mean']
        return datetime.timedelta(seconds=time.time() - self._epoch_start_time)

    @property
    def epoch_remaining_time(self):
        x = self.training_time_summary.stats(fields=['time'], backlog=20)
        x = x['time']['mean']
        remaining_batches = self.batches_per_epoch - self._batch_number
        return datetime.timedelta(seconds=remaining_batches * x)

    @property
    def elapsed_time(self):
        return datetime.timedelta(seconds=time.time() - self._training_start_time)

    @property
    def remaining_time(self):
        x = self.training_time_summary.stats(fields=['time'], backlog=20)
        x = x['time']['mean']
        remaining_batches = self._total_batches - self._batch_index
        return datetime.timedelta(seconds=remaining_batches * x)

    @property
    def epoch_percent(self):
        epoch_percent = float(self._batch_number) / float(self._batches_per_epoch)
        epoch_percent *= 100
        return epoch_percent

    @property
    def training_percent(self):
        epoch_percent = float(self._batch_index) / float(self._total_batches)
        epoch_percent *= 100
        return epoch_percent

    def get_average_training_loss(self, time_period=None):
        return self.train_summary.stats(fields=['loss'], backlog=time_period)['loss']['mean']

    def get_average_training_accuracy(self, time_period=None):
        return self.train_summary.stats(fields=['accuracy'], backlog=time_period)['accuracy']['mean']

    def get_average_validation_loss(self, time_period=None):
        return self.validation_summary.stats(fields=['loss'], backlog=time_period)['loss']['mean']

    def get_average_validation_accuracy(self, time_period=None):
        return self.validation_summary.stats(fields=['accuracy'], backlog=time_period)['accuracy']['mean']

    def __default_validation_iterator(self):
        while True:
            yield None

    def __float_dict(self, d):
        return {'accuracy': float(d['accuracy']),
                'loss': float(d['loss']),
                'output': d.get('output', None)}

    def train_on_batch(self, train_x, train_y):
        self.batch_index += 1
        d = self.model.train(train_x, train_y, session=self._session)
        d = self.__float_dict(d)
        self.train_summary.add(self.batch_index, **d)
        return d

    def validate_on_batch(self, train_x, train_y):
        d = self.model.validate(train_x, train_y, session=self._session)
        d = self.__float_dict(d)
        self.validation_summary.add(self.batch_index, **d)
        return d

    def test_on_batch(self, train_x, train_y):
        d = self.model.test(train_x, train_y, session=self._session)
        d = self.__float_dict(d)
        return d

    def get_loss_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.train_summary.get(fields=['loss'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        y = self.validation_summary.get(fields=['loss'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        return {'train': x['loss'], 'validation': y['loss']}

    def get_accuracy_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.train_summary.get(fields=['accuracy'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        y = self.validation_summary.get(fields=['accuracy'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        return {'train': x['accuracy'], 'validation': y['accuracy']}

    def _update_progress_info(self, training_batch):
        self._epoch_number = training_batch.get('epoch_number', None)
        self._total_epochs = training_batch.get('total_epochs', None)
        self._batch_number = training_batch.get('batch_number', None)
        self._batches_per_epoch = training_batch.get('batches_per_epoch', None)
        self._batch_index = training_batch.get('batch_index', None)
        self._total_batches = training_batch.get('total_batches', None)

    def __process_output(self, out):
        return [{"input": string,
                 'truth': int(truth),
                 'predicted': int(predicted)}
                for string, truth, predicted in out]

    def run_training(self, epochs=10, train_batch_size=100, validation_batch_size=100, test_batch_size=500):
        data_generator = self.data['constructor'](batch_size=train_batch_size, epochs=epochs, validation_size=validation_batch_size, test_size=test_batch_size)
        training_data_generator = data_generator['train']
        validation_iterator = data_generator['validation']  or self.__default_validation_iterator()
        test_iterator = data_generator['test']  or self.__default_validation_iterator()
        self._training_start_time = time.time()
        self._epoch_start_time = time.time()
        last_test_time = self._training_start_time
        current_epoch = 1
        test_index = 1
        for training_batch in training_data_generator:
            if training_batch['epoch_number'] != current_epoch:
                self._epoch_start_time = time.time()
            self._update_progress_info(training_batch)
            batch_t_0 = time.time()
            test_result_on_train = None
            if (self.test_interval is not None) and \
                    (time.time() - last_test_time >= self.test_interval):
                test_batch = next(test_iterator)
                result = None
                if test_batch is not None:
                    result = self.test_on_batch(train_x=test_batch['train_x'], train_y=test_batch['train_y'])
                    result['output'] = self.__process_output(result['output'])
                    last_test_time = time.time()
                    test_index += 1
                yield TestData(**result)

            d = self.train_on_batch(train_x=training_batch['train_x'], train_y=training_batch['train_y'])
            yield TrainData(**d)

            self.batch_index += 1

            if (self.validation_interval is not None) and \
                    ((training_batch['batch_index'] % self.validation_interval) == 0):
                validation_batch = next(validation_iterator)
                if validation_batch is not None:
                    d = self.validate_on_batch(train_x=validation_batch['train_x'], train_y=validation_batch['train_y'])
                    yield ValidationData(**d)
            batch_time = time.time() - batch_t_0
            self.training_time_summary.add(self.batch_index, time=batch_time)

#    def save_test(self, *args, **kwargs):
#        pass

#    def save_training_checkpoint(self, *args, **kwargs):
#        pass

#    def housekeeping(self):
#        pass
