import time
import os
import math
import json
import tensorflow as tf
import glob
from models.tf_session import tf_session
import datetime
from train.supervisor import TrainingSupervisor
from web.python.datasources import get_dataset_specs


class PersistentTrainingSupervisor(TrainingSupervisor):
    def __init__(self, model=None, **kwargs):
        super(PersistentTrainingSupervisor, self).__init__(model._model, **kwargs)
        # self._meta has the meta graph with the path info
        self._meta = model

    def save_test(self, train=None, test=None):
        test_root = self._meta.test_root
        test_result = {'train': train, 'test': test}
        test_confusion_result = self.make_test_output_matrix(train, test)
        test_file_name = "test-output-raw-{date}.json".format(date=datetime.datetime.today().isoformat())
        test_confusion_name = "test-output-matrix-{date}.json".format(date=datetime.datetime.today().isoformat())
        test_path = os.path.join(test_root, test_file_name)
        test_confusion_path = os.path.join(test_root, test_confusion_name)
        with open(test_path, 'w') as test_file:
            test_file.write(json.dumps(test_result))
        with open(test_confusion_path, 'w') as test_file:
            test_file.write(json.dumps(test_confusion_result))

        print('Save test')

    def save_training_checkpoint(self, file_name):
        return self._meta.save_training_state()

    def housekeeping(self):
        pass

    def __format_confusion_matrix(self, labels, true_labels, predicted_labels):
        matrix = {}
        for i in labels:
            for j in labels:
                matrix[i, j] = 0
        for t_l, p_l in zip(true_labels, predicted_labels):
            if (t_l, p_l) not in matrix:
                matrix[(t_l, p_l)] = 0
            matrix[(t_l, p_l)] += 1
        return [[i, j, matrix[i, j]] for i, j in matrix]

    def make_test_output_matrix(self, train, test):
        test_true_values = [x['truth'] for x in test['output']]
        test_predicted_values = [x['predicted'] for x in test['output']]
        test_confusion_matrix = self.__format_confusion_matrix([0, 1, 2], test_true_values, test_predicted_values)

        train_true_values = [x['truth'] for x in train['output']]
        train_predicted_values = [x['predicted'] for x in train['output']]
        train_confusion_matrix = self.__format_confusion_matrix([0, 1, 2], train_true_values, train_predicted_values)
        return {'train': {'loss': train['loss'],
                          'accuracy': train['accuracy'],
                          'matrix': train_confusion_matrix},
                'test': {'loss': test['loss'],
                         'accuracy': test['accuracy'],
                         'matrix': test_confusion_matrix}}

    def train_model(self, batch_size=100, validation_size=None, test_size=None, epochs=None):
        data = get_dataset_specs(self._meta.dataset)['constructor']
        data_generator = data(batch_size=batch_size,
                              epochs=epochs,
                              validation_size=validation_size,
                              test_size=test_size)
        self.run_training(data_generator['train'], data_generator['validation'], data_generator['test'],
                          session=self._meta._session)
