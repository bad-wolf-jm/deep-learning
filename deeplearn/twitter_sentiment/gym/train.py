import tensorflow as tf
import time
import os
from models.supervisor_2 import TrainingSupervisor
from models.supervisor_2 import TrainData, TestData, ValidationData
from models.test_compile import CompiledTrainingModel
from notify.send_mail import EmailNotification
from gym.webmon import render_template, start, post_test
import traceback


model_path = 'models/bigru_3.py'
model_name = os.path.basename(model_path)
model_dir = os.path.dirname(model_path)
train_dir = os.path.join(model_dir, '.train', model_name)
weight_dir = os.path.join(train_dir, 'weights')
log_dir = os.path.join(train_dir, 'logs')
weight_file = os.path.join(weight_dir, model_name)

for d in [train_dir, weight_dir, log_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

MINUTES = 60
HOURS = 3600
SECONDS = 1

validation_interval = 5
test_interval = 2 * MINUTES
e_mail_interval = 1.5 * HOURS
summary_span = None
checkpoint_interval = 45 * MINUTES
train_batch_size = 100
validation_batch_size = 100
test_batch_size = 1000
epochs = 50


tests = []

_last_email_time = time.time()
_last_checkpoint_time = time.time()


def __format_confusion_matrix(labels, true_labels, predicted_labels):
    matrix = {}
    for i in labels:
        for j in labels:
            matrix[i, j] = 0
    for t_l, p_l in zip(true_labels, predicted_labels):
        if (t_l, p_l) not in matrix:
            matrix[(p_l, t_l)] = 0
        matrix[(p_l, t_l)] += 1
    return matrix


def make_test_output_matrix(test):
    labels = sorted(supervisor.model.categories.keys())
    test_true_values = [x['truth'] for x in test.output]
    test_predicted_values = [x['predicted'] for x in test.output]
    test_confusion_matrix = __format_confusion_matrix(labels, test_true_values, test_predicted_values)
    return {'loss': test.loss,
            'accuracy': test.accuracy,
            'result': test.output,
            'matrix': test_confusion_matrix}


def send_report_email():
    global _last_email_time
    test_matrices = []
    for test_data in tests:
        matrix = make_test_output_matrix(test_data)
        test_matrices.append(matrix)
    _last_email_time = time.time()
    x = render_template('email.html', test_matrices=test_matrices, supervisor=supervisor)
    EmailNotification.sendEmail(x, subject="Training report")


def save_checkpoint():
    x.save(weight_file, session=_session)


x = CompiledTrainingModel(model_path)
with tf.Session(graph=x._graph) as _session:
    x.initialize(_session)
    supervisor = TrainingSupervisor(session=_session, model=x,
                                    test_interval=test_interval,
                                    validation_interval=validation_interval,
                                    summary_span=summary_span)
    start(supervisor)
    for loss in supervisor.run_training(epochs=epochs,
                                        train_batch_size=train_batch_size,
                                        validation_batch_size=validation_batch_size,
                                        test_batch_size=test_batch_size):
        try:
            if isinstance(loss, TestData):
                tests.append(loss)
                post_test(loss)
            current_time = time.time()
            time_since_last_email = current_time - _last_email_time
            time_since_last_checkpoint = current_time - _last_checkpoint_time
            if time_since_last_email > e_mail_interval:
                send_report_email()
                _last_email_time = time.time()
            if time_since_last_checkpoint > checkpoint_interval:
                save_checkpoint()
                _last_checkpoint_time = time.time()
        except Exception as error:
            traceback.print_exc()
            print('ERROR', error)
if __name__ == '__main__':
    sys.exit(0)
