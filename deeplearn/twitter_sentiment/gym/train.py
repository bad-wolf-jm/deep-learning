import tensorflow as tf
import time
import os
from models.supervisor_2 import TrainingSupervisor
from models.supervisor_2 import TrainData, TestData, ValidationData
from models.test_compile import CompiledTrainingModel
from notify.send_mail import EmailNotification
from gym.webmon import render_template, start, post_test


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

validation_interval = 5
test_interval = 5 * 60
e_mail_interval = 1.5 * 3600
summary_span = None
checkpoint_interval = 45 * 60
train_batch_size = 100
validation_batch_size = 100
test_batch_size = 1000
epochs = 50


tests = []

_last_email_time = time.time()
_last_checkpoint_time = time.time()


def send_report_email():
    def __matrix_to_dict(ll):
        return {(i, j): n for i, j, n in ll}
    foo = supervisor._meta.get_confusion_matrices(min_date=last_email_time)
    bar = [json.loads(open(x).read()) for x in foo]
    test_matrices = []
    for file_path in foo:
        matrix = json.loads(open(file_path).read())
        test_time = datetime.datetime.fromtimestamp(os.stat(file_path).st_ctime)
        t = matrix['test']
        t['time'] = test_time.isoformat()
        t['matrix'] = __matrix_to_dict(t['matrix'])
        test_matrices.append(t)
    last_email_time = time.time()
    x = render_template('email.html', test_matrices=test_matrices, supervisor=supervisor)
    EmailNotification.sendEmail(x, subject="Training report")


def save_checkpoint():
    x.save(_session, weight_file)


x = CompiledTrainingModel('models/bigru_3.py')
with tf.Session(graph=x._graph) as _session:
    x.initialize(_session)
    supervisor = TrainingSupervisor(session=_session, model=x, test_interval=test_interval, validation_interval=validation_interval, summary_span=summary_span)
    start(supervisor)
    for loss in supervisor.run_training(epochs=epochs, train_batch_size=train_batch_size, validation_batch_size=validation_batch_size, test_batch_size=test_batch_size):
        try:
            current_time = time.time()
            time_since_last_email = current_time - _last_email_time
            time_since_last_checkpoint = current_time - _last_checkpoint_time
            if time_since_last_email > e_mail_interval:
                send_report_email()
            if time_since_last_checkpoint > checkpoint_interval:
                save_checkpoint()
            if isinstance(loss, TestData):
                tests.append(loss)
                post_test(loss)
        except:
            print('ERROR')
if __name__ == '__main__':
    sys.exit(0)
