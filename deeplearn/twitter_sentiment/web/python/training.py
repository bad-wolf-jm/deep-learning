import time
import os
import math
import json
import numpy as np
import tensorflow as tf
import glob
from graphs.base.tf_session import tf_session
import datetime
import threading
from train import optimizers
from train.supervisor import TrainingSupervisor
from train.datasources import get_dataset_specs
#from web.python.bootstrap import PersistentGraph


class PersistentTrainingSupervisor(TrainingSupervisor):
    def __init__(self, model=None, optimizer=None, batch_size=None, validation_size=None, test_size=None,
                 epochs=None, e_mail_interval=None, **kwargs):
        super(PersistentTrainingSupervisor, self).__init__(model._model, **kwargs)
        self._meta = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.epochs = epochs
        self.e_mail_interval = e_mail_interval
        self.test_index = 0


        with self._meta._graph.as_default():
            optimizer_id = self.optimizer['name']
            optimizer = optimizers.get_by_id(optimizer_id)
            self.optimizer['display_name'] = optimizer['display_name']
            optimizer['learning_rate'] = self.optimizer['learning_rate']
            optimizer['optimizer_parameters'].update(self.optimizer['optimizer_parameters'])
            self._meta._model.train_setup(optimizer['constructor'],
                                          optimizer['learning_rate'],
                                          **optimizer['optimizer_parameters'])

    def make_test_output_matrix(self, train, test):
        labels = sorted(self._meta.category_labels.keys())
        test_true_values = [x['truth'] for x in test['output']]
        test_predicted_values = [x['predicted'] for x in test['output']]
        test_confusion_matrix = self.__format_confusion_matrix(labels, test_true_values, test_predicted_values)

        train_true_values = [x['truth'] for x in train['output']]
        train_predicted_values = [x['predicted'] for x in train['output']]
        train_confusion_matrix = self.__format_confusion_matrix(labels, train_true_values, train_predicted_values)
        return {'train': {'loss': train['loss'],
                          'accuracy': train['accuracy'],
                          'matrix': train_confusion_matrix},
                'test': {'loss': test['loss'],
                         'accuracy': test['accuracy'],
                         'matrix': test_confusion_matrix}}

    def save_test(self, train=None, test=None):
        test_root = self._meta.test_root
        test_result = {'train': train, 'test': test}
        test_confusion_result = self.make_test_output_matrix(train, test)
        test_tag_name = "test-tag-{date}.json".format(date=datetime.datetime.today().isoformat())
        test_file_name = "test-output-raw-{date}.json".format(date=datetime.datetime.today().isoformat())
        test_confusion_name = "test-output-matrix-{date}.json".format(date=datetime.datetime.today().isoformat())
        test_path = os.path.join(test_root, test_file_name)
        test_tag_path = os.path.join(test_root, test_tag_name)
        test_confusion_path = os.path.join(test_root, test_confusion_name)
        with open(test_path, 'w') as test_file:
            test_file.write(json.dumps(test_result))
        with open(test_confusion_path, 'w') as test_file:
            test_file.write(json.dumps(test_confusion_result))
        test_pack = {'loss': test['loss'],
                     'accuracy': test['accuracy'],
                     'date': time.time(),
                     'confusion_file': test_confusion_path,
                     'output_file': test_path}
        with open(test_tag_path, 'w') as test_file:
            test_file.write(json.dumps(test_pack))
        # also save loss and accuracy graphs

    def save_training_checkpoint(self):
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
        return [[i, j, matrix[j, i]] for i, j in matrix]

    def __is_nan_of_infinite(self, num):
        return (np.isnan([num]) or np.isinf([num]))

    def do_train_model(self):
        data = self._meta.dataset['constructor']
        data_generator = data(batch_size=self.batch_size,
                              epochs=self.epochs,
                              validation_size=self.validation_size,
                              test_size=self.test_size)
        train_loop = self.run_training(data_generator['train'], data_generator['validation'], data_generator['test'],
                                       session=self._meta._session)
        for train_loss in train_loop:
            yield train_loss

    def half_learning_rate(self):
        learning_rate = self._meta._model.learning_rate / 10.0
        self._meta.build()
        self._meta.initialize()
        self._meta.restore_last_checkpoint()
        with self._meta._graph.as_default():
            self._meta._model.train_setup(tf.train.AdamOptimizer, learning_rate=learning_rate)
            self._meta.initialize_uninitialized_variables()
        self._session = self._meta._session
        self.model = self._meta._model
        pass


class ThreadedModelTrainer(object):
    def __init__(self, model_graph=None, train_settings=None): #, initial_weights=None):
        super(ThreadedModelTrainer, self).__init__()
        self.model_graph = model_graph
        self.train_settings = train_settings
        #self.initial_weights = initial_weights
        self.training_supervisor = None
        self.is_running = False
        self.ready_lock = threading.Lock()
        self.ready_lock.acquire()
        self.__internal_thread = None

        # print(self.model_graph)
        # Keep an internal /tmp folder to save models before stopping
        # if there is one in there, resume training
        # add a 'reset' function, which deletes the saved image, and
        # forces the training to reinitialize everything.

    def __is_nan_of_infinite(self, num):
        return (np.isnan([num]) or np.isinf([num]))

    def prefix_exists(self, prefix):
        return len(glob.glob("{p}*".format(p=prefix))) > 0

    def run(self):
        self.is_running = True
        self.model_graph.build(training=True)
        self.training_supervisor = PersistentTrainingSupervisor(self.model_graph, **self.train_settings)
        self.model_graph.initialize()
        self.ready_lock.release()
        for training_loss in self.training_supervisor.do_train_model():
            if self.__is_nan_of_infinite(training_loss):
                self.training_supervisor.half_learning_rate()
                print("rate lowered... using new learning rate", self.model_graph._model.learning_rate)
                print("using new learning rate")
            if not self.is_running:
                break
            print(training_loss)

    def start(self, initial_weights=None):
        if self.__internal_thread is None:
            self.__internal_thread = threading.Thread(target=self.run)
            self.__internal_thread.start()
            self.ready_lock.acquire()

    def stop(self):
        if self.__internal_thread is not None:
            self.is_running = False
            self.__internal_thread.join()
            self.__internal_thread = None


#class ThreadedModelTrainer_XXX(object):
#    def __init__(self, model_graph=None, train_settings=None): #, initial_weights=None):
#        super(ThreadedModelTrainer, self).__init__()
#        self.model_graph = model_graph
#        self.train_settings = train_settings
#        #self.initial_weights = initial_weights
#        self.training_supervisor = None
#        self.is_running = False
#        self.ready_lock = threading.Lock()
#        self.ready_lock.acquire()
#        self.__internal_thread = None
#
#        # print(self.model_graph)
#        # Keep an internal /tmp folder to save models before stopping
#        # if there is one in there, resume training
#        # add a 'reset' function, which deletes the saved image, and
#        # forces the training to reinitialize everything.
#
#    def __is_nan_of_infinite(self, num):
#        return (np.isnan([num]) or np.isinf([num]))
#
#    def prefix_exists(self, prefix):
#        return len(glob.glob("{p}*".format(p=prefix))) > 0
#
#    def run(self):
#        self.is_running = True
#        self.model_graph.build(training=True)
#        self.training_supervisor = PersistentTrainingSupervisor(self.model_graph, **self.train_settings)
#        self.model_graph.initialize()
#        self.ready_lock.release()
#        for training_loss in self.training_supervisor.do_train_model():
#            if self.__is_nan_of_infinite(training_loss):
#                self.training_supervisor.half_learning_rate()
#                print("rate lowered... using new learning rate", self.model_graph._model.learning_rate)
#                print("using new learning rate")
#            if not self.is_running:
#                break
#            print(training_loss)
#
#    def start(self, initial_weights=None):
#        if self.__internal_thread is None:
#            self.__internal_thread = threading.Thread(target=self.run)
#            self.__internal_thread.start()
#            self.ready_lock.acquire()
#
#    def stop(self):
#        if self.__internal_thread is not None:
#            self.is_running = False
#            self.__internal_thread.join()
#            self.__internal_thread = None
