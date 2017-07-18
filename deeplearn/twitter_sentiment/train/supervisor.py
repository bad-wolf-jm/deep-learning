import time
import os
import math
import json
import tensorflow as tf
import glob
from models.tf_session import tf_session
from train.summary import TrainingSummary
import datetime


class TrainingSupervisor(object):
    def __init__(self, model, validation_interval=None, test_interval=None, summary_span=None, checkpoint_interval=None,
                 test_keep=None, checkpoint_keep=None):
        super(TrainingSupervisor).__init__()
        self.model = model
        self.batch_index = 0

        _model_data_root = '~/.sentiment_analysis/training/{model_name}'.format(model_name=type(self.model).__name__)
        _model_data_root = os.path.expanduser(_model_data_root)
        _model_data_files = ['test', 'checkpoints', 'metadata']

        for p in _model_data_files:
            _x = os.path.join(_model_data_root, p)
            if not os.path.exists(_x):
                os.makedirs(_x)

        self.validation_interval = validation_interval
        self.checkpoint_interval = checkpoint_interval  # in batches
        self.checkpoint_keep = checkpoint_keep
        self.test_interval = 1  # test_interval  # in seconds
        self.test_keep = test_keep

        self.checkpoint_root = os.path.join(_model_data_root, 'checkpoints')
        self.test_root = os.path.join(_model_data_root, 'test')
        self.metadata_root = os.path.join(_model_data_root, 'metadata')

        self.summary = TrainingSummary(summary_span=None, fields=['loss', 'accuracy', 'time'])

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
             'time': float(d['time']),
             'output': d['output']}
        return d

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

    def _clean_test_folder(self):
        files = [[f, os.stat(f).st_ctime]for f in glob.glob("{root}/*.json".format(root=self.test_root))]
        files = sorted(files, key=lambda x: x[1], reverse=True)[self.test_keep or 10:]
        for f in files:
            os.unlink(f[0])

    def get_test_results(self):
        files = [[f, os.stat(f).st_ctime]for f in glob.glob("{root}/*.json".format(root=self.test_root))]
        files = sorted(files, key=lambda x: x[1], reverse=True)  # [self.test_keep or 10:]
        return [f[0] for f in files]

    def run_training(self, training_data_generator, validation_data_generator=None, resume_from_checkpoint=None):
        validation_iterator = validation_data_generator or self.__default_validation_iterator()
        tf_session().run(tf.global_variables_initializer())
        if resume_from_checkpoint is not None:
            _p = os.path.join(self.checkpoint_root, resume_from_checkpoint + '.chkpt')
            if os.path.exists(_p):
                saver = tf.train.Saver()
                saver.restore(tf_session(), _p)
        self._training_start_time = time.time()
        last_test_time = self._training_start_time
        test_index = 1
        for training_batch in training_data_generator:
            self._update_progress_info(training_batch)
            test_result_on_train = None
            if (self.test_interval is not None) and (time.time() - last_test_time >= self.test_interval):
                test_result_on_train = self.test_on_batch(train_x=training_batch['train_x'], train_y=training_batch['train_y'])
                test_result_on_train['output'] = [{'string': string, 'truth': int(truth), 'predicted': int(predicted)}
                                                  for string, truth, predicted in test_result_on_train['output']]
                validation_batch = next(validation_iterator)
                if validation_batch is not None:
                    result = self.test_on_batch(train_x=validation_batch['train_x'],
                                                train_y=validation_batch['train_y'])
                    result_output = result['output']
                    test_output_file = "{model_name}-test-{index}-loss:{loss:.4f}-accuracy:{accuracy:.2f}.json"
                    test_output_file = test_output_file.format(model_name=type(self.model).__name__,
                                                               index=test_index, loss=result['loss'],
                                                               accuracy=100 * result['accuracy'])
                    test_output_path = os.path.join(self.test_root, test_output_file)
                    result['output'] = [{'string': string, 'truth': int(truth), 'predicted': int(predicted)}
                                        for string, truth, predicted in result_output]
                    output_string = json.dumps({'train': test_result_on_train,
                                                'test': result})
                    with open(test_output_path, 'w') as to_file:
                        to_file.write(output_string)
                    last_test_time = time.time()
                    test_index += 1
                    self._clean_test_folder()

            self.train_on_batch(train_x=training_batch['train_x'], train_y=training_batch['train_y'])
            self.batch_index += 1

            if (self.checkpoint_interval is not None) and ((training_batch['batch_index'] % self.checkpoint_interval) == 0):
                path = self.save_training_checkpoint('training-checkpoint.chkpt')
                print('Saving checkpoint:', path)

            if (self.validation_interval is not None) and ((training_batch['batch_index'] % self.validation_interval) == 0):
                validation_batch = next(validation_iterator)
                if validation_batch is not None:
                    self.validate_on_batch(train_x=validation_batch['train_x'], train_y=validation_batch['train_y'])

    def save_model_as(self, dirname, file_name):
        saver = tf.train.Saver()
        checkpoint_name = file_name
        x = os.path.join(dirname, checkpoint_name)
        path = saver.save(tf_session(), x)
        return path

    def save_training_checkpoint(self, file_name):
        return self.save_model_as(self.checkpoint_root, file_name)

    def save_model_image(self, *a):
        return self.save_model_as('restore-model-image.chkpt')

    def shutdown(self):
        pass
