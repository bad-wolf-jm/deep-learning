import time
import os
import math
import tensorflow as tf
from models.tf_session import tf_session
from train.summary import TrainingSummary
import datetime
#from stream.sender import DataStreamer
#from stream.receiver import DataReceiver


class TrainingSupervisor(object):
    def __init__(self, model, validation_interval=None, test_interval=None, summary_span=None):
        super(TrainingSupervisor).__init__()
        self.model = model
        self.batch_index = 0
        self.checkpoint_interval = 250
        self.checkpoint_keep = 10
        _r = os.path.expanduser('~/.sentiment_analysis/checkpoints/')
        _r = os.path.join(_r, type(model).__name__)
        self.checkpoint_root = _r
        if not os.path.exists(self.checkpoint_root):
            os.makedirs(self.checkpoint_root)
        self.summary = TrainingSummary(summary_span=None, fields=['loss', 'accuracy', 'time'])

        self.validation_interval = validation_interval
        self.test_interval = validation_interval
        self.batch_index = 0
        self._epoch_number = None
        self._total_epochs = None
        self._batch_number = None
        self._batches_per_epoch = None
        self._batch_index = None
        self._total_batches = None

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
        x = self.summary.get_stats('train', fields=['time'], backlog=10)
        x = x['time']['mean']
        return datetime.timedelta(seconds=x)

    @property
    def epoch_time(self):
        x = self.summary.get_stats('train', fields=['time'], backlog=10)
        x = x['time']['mean']
        return datetime.timedelta(seconds=self.batches_per_epoch * x)

    @property
    def elapsed_time(self):
        return datetime.timedelta(seconds=time.time() - self._training_start_time)

    @property
    def remaining_time(self):
        x = self.summary.get_stats('train', fields=['time'], backlog=10)
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
        return self.summary.get_stats('train', fields=['loss'], backlog=time_period)['loss']['mean']

    def get_average_training_accuracy(self, time_period=None):
        return self.summary.get_stats('train', fields=['accuracy'], backlog=time_period)['accuracy']['mean']

    def get_average_validation_loss(self, time_period=None):
        return self.summary.get_stats('validation', fields=['loss'], backlog=time_period)['loss']['mean']

    def get_average_validation_accuracy(self, time_period=None):
        return self.summary.get_stats('validation', fields=['accuracy'], backlog=time_period)['accuracy']['mean']

    def __default_validation_iterator(self):
        while True:
            yield None

    def train_on_batch(self, train_x, train_y):
        self.batch_index += 1
        d = self.train_step(train_x, train_y)
        d = {'accuracy': float(d['accuracy']),
             'loss': float(d['loss']),
             'time': float(d['time'])}
        self.summary.add_to_summary('train', self.batch_index, **d)
        if (self.batch_index % self.checkpoint_interval) == 0:
            path = self.save_model_as('training-checkpoint.chkpt')
            print('Saving checkpoint:', path)
        return d

    def validate_on_batch(self, train_x, train_y):
        d = self.validation_step(train_x, train_y)
        d = {'accuracy': float(d['accuracy']),
             'loss': float(d['loss']),
             'time': float(d['time'])}
        self.summary.add_to_summary('validation', self.batch_index, **d)
        return d

    def test_on_batch(self, train_x, train_y):
        d = self.test_model(train_x, train_y)
        d = {'accuracy': float(d['accuracy']),
             'loss': float(d['loss']),
             'time': float(d['time'])}
        return d

    # def get_train_summary(self, min_batch_index=None, max_batch_index=None):
    #    x = self.summary.get_summary('train', min_batch_index=min_batch_index, max_batch_index=max_batch_index)
    #    x['epoch_number'] = self._epoch_number
    #    x['total_epochs'] = self._total_epochs
    #    x['batch_number'] = self._batch_number
    #    x['batches_per_epoch'] = self._batches_per_epoch
    #    x['batch_index'] = self._batch_index
    #    x['total_batches'] = self._total_batches
    #    return x

    # def get_validation_summary(self, min_batch_index=None, max_batch_index=None):
    #    x = self.summary.get_summary('validation', min_batch_index=min_batch_index, max_batch_index=max_batch_index)
    #    return x

    def get_loss_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.summary.get_summary('train', fields=['loss'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        y = self.summary.get_summary('validation', fields=['loss'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        return {'train': x['loss'],
                'validation': y['loss']}

    def get_accuracy_summary(self, min_batch_index=None, max_batch_index=None):
        x = self.summary.get_summary('train', fields=['accuracy'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        y = self.summary.get_summary('validation', fields=['accuracy'], min_batch_index=min_batch_index, max_batch_index=max_batch_index)
        return {'train': x['accuracy'],
                'validation': y['accuracy']}

    def _update_progress_info(self, training_batch):
        self._epoch_number = training_batch.get('epoch_number', None)
        self._total_epochs = training_batch.get('total_epochs', None)
        self._batch_number = training_batch.get('batch_number', None)
        self._batches_per_epoch = training_batch.get('batches_per_epoch', None)
        self._batch_index = training_batch.get('batch_index', None)
        self._total_batches = training_batch.get('total_batches', None)

    def run_training(self, training_data_generator, validation_data_generator=None, resume_from_checkpoint=None):
        validation_iterator = validation_data_generator or self.__default_validation_iterator()
        tf_session().run(tf.global_variables_initializer())
        if resume_from_checkpoint is not None:
            _p = os.path.join(self.checkpoint_root, resume_from_checkpoint + '.chkpt')
            if os.path.exists(_p):
                saver = tf.train.Saver()
                saver.restore(tf_session(), _p)
        self._training_start_time = time.time()
        for training_batch in training_data_generator:
            self._update_progress_info(training_batch)
            self.train_on_batch(train_x=training_batch['train_x'], train_y=training_batch['train_y'])
            self.batch_index += 1
            if training_batch['batch_index'] % self.validation_interval == 0:
                validation_batch = next(validation_iterator)
                if validation_batch is not None:
                    self.validate_on_batch(train_x=validation_batch['train_x'], train_y=validation_batch['train_y'])
            if training_batch['batch_index'] % self.test_interval == 0:
                validation_batch = next(validation_iterator)
                if validation_batch is not None:
                    self.test_on_batch(train_x=validation_batch['train_x'], train_y=validation_batch['train_y'])

    def save_model_as(self, file_name):
        saver = tf.train.Saver()
        checkpoint_name = file_name
        x = os.path.join(self.checkpoint_root, checkpoint_name)
        path = saver.save(tf_session(), x)
        return path

    def save_model_image(self, *a):
        return self.save_model_as('restore-model-image.chkpt')

    def shutdown(self):
        pass
