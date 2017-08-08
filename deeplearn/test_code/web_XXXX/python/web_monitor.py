import json
import os
import psutil
from web.python.training import ThreadedModelTrainer


class JSONReturnValues(object):
    def __init__(self):
        super(JSONReturnValues, self).__init__()

    def status_ok(self, value):
        return json.dumps({'status': 'ok', 'value': value, 'message': None})

    def status_error(self, value):
        return json.dumps({'status': 'error', 'message': value, 'value': None})


class SystemMonitor(JSONReturnValues):
    def __init__(self):
        super(TrainingMonitor, self).__init__()

    def get_system_info(self):
        memory = psutil.virtual_memory()
        used = memory.total - memory.available
        return self.status_ok({'cpu': psutil.cpu_percent(),
                               'memory': [used, memory.total]})


class TrainingMonitor(JSONReturnValues):
    def __init__(self):
        super(TrainingMonitor, self).__init__()
        self._training_thread = None
        self._training_supervisor = None
        self._stats_lookback = 15

    def get_training_statistics(self):
        supervisor = self._training_supervisor
        if supervisor is not None:
            stats = {
                'training': {
                    'loss': supervisor.get_average_training_loss(self._stats_lookback),
                    'accuracy': supervisor.get_average_training_accuracy(self._stats_lookback)
                },
                'validation': {
                    'loss': supervisor.get_average_validation_loss(self._stats_lookback),
                    'accuracy': supervisor.get_average_validation_accuracy(self._stats_lookback)
                }
            }
            return self.status_ok(stats)
        else:
            return self.status_error('No training in progress')

    def get_latest_confusion_matrix(self):
        supervisor = self._training_supervisor
        if supervisor is not None:
            files = supervisor._meta.get_confusion_matrices()
            if len(files) > 0:
                f = files[0]
                with open(f) as matrix_file:
                    x = json.loads(matrix_file.read())
                    return self.status_ok(x)
        return self.status_error('No training in progress')

    def get_confusion_matrices(self, min_date=None, max_date=None):
        supervisor = self._training_supervisor
        if supervisor is not None:
            files = supervisor._meta.get_confusion_matrices(min_date, max_date)
            res = []
            for matrix_file in files:
                with open(matrix_file) as file_:
                    x = json.loads(file_.read())
                    res.append(x)
            return self.status_ok(res)
        return self.status_error('No training in progress')

    def get_training_progress(self):
        supervisor = self._training_supervisor
        if supervisor is not None:
            progress = {'batch_number': supervisor.batch_number,
                        'batches_per_epoch': supervisor.batches_per_epoch,
                        'epoch_number': supervisor.epoch_number,
                        'percent_epoch_complete': supervisor.epoch_percent,
                        'percent_training_complete': supervisor.training_percent,
                        'total_epochs': supervisor.number_of_epochs,
                        'batch_time': supervisor.batch_time.total_seconds(),
                        'epoch_time': supervisor.epoch_time.total_seconds(),
                        'elapsed_time': supervisor.elapsed_time.total_seconds(),
                        'remaining_time': supervisor.remaining_time.total_seconds()}
            return self.status_ok(progress)
        return self.status_error('No training in progress')

    def get_training_graph_series(self, min_timestamp=None, max_timestamp=None):
        supervisor = self._training_supervisor
        if supervisor is not None:
            loss = supervisor.get_loss_summary(min_timestamp, max_timestamp)
            accuracy = supervisor.get_accuracy_summary(min_timestamp, max_timestamp)
            return self.status_ok({'loss': loss, 'accuracy': accuracy})
        else:
            return self.status_error("No training in progress")

    def load_model(self, model_type=None, model_name=None, train_settings=None):
        self._training_thread = ThreadedModelTrainer(model_name=model_name, model_type=model_type, train_settings=train_settings)
        return self.status_ok("Model succesfully loaded")

    def stop_training(self):
        if self._training_thread is not None:
            self._training_thread.stop()
            self._training_thread = None
            self._training_supervisor = None
            return self.status_ok("Training stopped successfully")
        return self.status_error('No training in progress')

    def start_training(self):
        if self._training_thread is not None:
            self._training_thread.start()
            return self.status_ok("Training started successfully")
        return self.status_error('No training in progress')
