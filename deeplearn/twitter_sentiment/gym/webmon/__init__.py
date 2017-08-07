from flask import Flask, Response, request
from flask_cors import CORS
import jinja2
import json
import os
#import re
#import random
#import time
from threading import Lock
import psutil
#import glob
#import datetime
import threading
import logging
import socket
import tensorflow as tf


log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)


#from web.python import bootstrap_yaml as bootstrap
# from web.python.bootstrap_yaml import CompiledModel  # , list_model_types
#from web.python.training import PersistentTrainingSupervisor, ThreadedModelTrainer
#from notify.send_mail import EmailNotification


D = os.path.dirname(__file__)  # os.path.dirname(os.path.dirname(__file__)))
print(__file__)
print(D)
# sys.exit(0)

TEMPLATES_ROOT = D

# os.path.join(os.path.expanduser('~'),
#                              'python',
#                              #'Python', 'DJ',
#                              'deep-learning',
#                              'deeplearn',
#                              'twitter_sentiment', 'web')

loader = jinja2.FileSystemLoader(
    [os.path.join(TEMPLATES_ROOT, "static"),
     os.path.join(TEMPLATES_ROOT, "templates")])
environment = jinja2.Environment(loader=loader)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)


def render_template(template_filename, **context):
    return environment.get_template(template_filename).render(context)


@app.route('/status/system_stats.json')
def get_system_info():
    memory = psutil.virtual_memory()
    used = memory.total - memory.available
    return json.dumps({'cpu': psutil.cpu_percent(),
                       'memory': [used, memory.total]})


@app.route('/json/training_stats.json')
def get_training_info():
    return json.dumps({'training': {'loss': supervisor.get_average_training_loss(15),
                                    'accuracy': supervisor.get_average_training_accuracy(15)},
                       'validation': {'loss': supervisor.get_average_validation_loss(15),
                                      'accuracy': supervisor.get_average_validation_accuracy(15)}})


def __format_confusion_matrix(labels, true_labels, predicted_labels):
    matrix = {}
    for i in labels:
        for j in labels:
            matrix[i, j] = 0
    for t_l, p_l in zip(true_labels, predicted_labels):
        if (t_l, p_l) not in matrix:
            matrix[(t_l, p_l)] = 0
        matrix[(t_l, p_l)] += 1
    return [[i, j, matrix[j, i]] for i, j in matrix]


def make_test_output_matrix(test):
    labels = sorted(supervisor.model.categories.keys())
    test_true_values = [x['truth'] for x in test.output]
    test_predicted_values = [x['predicted'] for x in test.output]
    test_confusion_matrix = __format_confusion_matrix(labels, test_true_values, test_predicted_values)
    return {'loss': test.loss,
            'accuracy': test.accuracy,
            'result': test.output,
            'matrix': test_confusion_matrix}


@app.route('/json/latest_test.json')
def get_latest_test():
    global latest_test
    #print (latest_test)
    with latest_test_lock:
        _ = make_test_output_matrix(latest_test)
    return json.dumps(_)


def get_training_status_struct():
    return {#'optimizer': optimizers[type(supervisor.model._optimizer)] if supervisor is not None else None,
            #'learning_rate': supervisor.model._learning_rate if supervisor is not None else None,
            'batch_number': supervisor.batch_number if supervisor is not None else 0,
            'batches_per_epoch': supervisor.batches_per_epoch if supervisor is not None else 0,
            'epoch_number': supervisor.epoch_number if supervisor is not None else 0,
            'percent_epoch_complete': supervisor.epoch_percent if supervisor is not None else 0,
            'percent_training_complete': supervisor.training_percent if supervisor is not None else 0,
            'total_epochs': supervisor.number_of_epochs if supervisor is not None else 0,
            'batch_time': supervisor.batch_time.total_seconds() if supervisor is not None else 0,
            'epoch_time': supervisor.epoch_time.total_seconds() if supervisor is not None else 0,
            'epoch_elapsed_time': supervisor.epoch_elapsed_time.total_seconds() if supervisor is not None else 0,
            'epoch_remaining_time': supervisor.epoch_remaining_time.total_seconds() if supervisor is not None else 0,
            'elapsed_time': supervisor.elapsed_time.total_seconds() if supervisor is not None else 0,
            'remaining_time': supervisor.remaining_time.total_seconds() if supervisor is not None else 0}


@app.route('/json/training_status.json')
def get_training_status():
    return json.dumps(get_training_status_struct())


@app.route('/fs/<path:path>')
def get_path(path):
    return open('/' + path).read()


@app.route('/json/training_graphs.json')
def get_training_graph_series():
    try:
        min_timestamp = float(request.args.get('min_timestamp'))
    except:
        min_timestamp = None
    try:
        max_timestamp = float(request.args.get('max_timestamp'))
    except:
        max_timestamp = None

    loss = supervisor.get_loss_summary(min_timestamp, max_timestamp)
    accuracy = supervisor.get_accuracy_summary(min_timestamp, max_timestamp)
    #print ('XXX')
    return json.dumps({'loss': loss,
                       'accuracy': accuracy})


@app.route('/static/<string:page_folder>/<path:page_name>')
def get_page(page_folder, page_name):
    dir_name = TEMPLATES_ROOT
    page_path = os.path.join(dir_name, 'static', page_folder, page_name)
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript"
    }

    ext = os.path.splitext(page_name)[1]
    mimetype = mimetypes.get(ext, "application/octet-stream")

    print(page_path, mimetype)
    try:
        return Response(open(page_path).read(), mimetype=mimetype)
    except Exception as e:
        print(e)
        return app.send_static_file(page_path)


@app.route('/json/tests.json')
def list_tests(min_date=None, max_date=None):
    test_list = supervisor._meta.get_tests()
    list_ = []
    for file_ in test_list:
        x = json.loads(open(file_).read())
        list_.append(x)
    return json.dumps(list_)


@app.route('/ui/training')
def display_training_status():
    template = 'nn_training.html'
    model_types = []  # bootstrap.list_model_types()
    return render_template(template,
                           host=socket.gethostname(),
                           initial_state=get_training_status_struct(),
                           supervisor=supervisor,
                           model_types=model_types)


#@app.route('/action/stop_training')
#def stop_training():
#    training_thread.stop()
#    print('TRAINING STOPPED')
#    return json.dumps({'status': 'ok'})


#@app.route('/action/start_training')
#def start_training():
#    training_thread.start()
#    print('TRAINING STARTED')
#    return json.dumps({'status': 'ok'})


# def send_email_every_minute():
#    while True:
#        print('Sending email')
#        try:
#            EmailNotification.sendEmail(get_report_email(), subject="Training report")
#            time.sleep(3600)
#        except:
#            time.sleep(6)


supervisor = None
latest_test = None
latest_test_lock = threading.Lock()
#host = os.environ['HOSTNAME']

# train_settings = {
#    'optimizer': {
#        'name': 'adam',
#        'learning_rate': 0.001,
#        'optimizer_parameters': {
#                'beta1': 0.9,
#                'beta2': 0.999,
#                'epsilon': 0.00000001
#        }
#    },
#    'validation_interval': 5,
#    'test_interval': 1 * 60,
#    'e_mail_interval': 1.5 * 3600,
#    'summary_span': None,
#    'checkpoint_interval': 45 * 60,
#    'batch_size': 100,
#    'validation_size': 100,
#    'test_size': 1000,
#    'epochs': 50
#}


def post_test(test_data):
    global latest_test
    print(test_data)
    with latest_test_lock:
        latest_test = test_data


def monitor():
    app.run(host='0.0.0.0', port=5000)


monitor_thread = threading.Thread(target=monitor)


def start(supervisor_struct):
    global supervisor
    supervisor = supervisor_struct
    monitor_thread.start()

# if __name__ == '__main__':
##    fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/tests/test2.yaml')
#    #fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/bigru_cms_user_3.yaml')
#    fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/bigru_cms_user_2.yaml')
#
#    #fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/gru_cms_user_2.yaml')
#    #fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/gru_conv_cms_user_2.yaml')
##    fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/tests/test2.yaml')
#
#    # model_name = ['Model_Tweet2Vec_BiGRU_CMSDataset',
#    #              'Model_Tweet2Vec_BiGRU_UserCMSDataset',
#    #              'Model_Tweet2Vec_BiGRU_BuzzometerDatasetVader']
#    #model_name = model_name[1]
#    #model_type = 'Tweet2Vec_BiGRU'
#    model_graph = CompiledModel.load_yaml(fil)
#    model_saved_settings = model_graph.yaml_train_settings
#    model_saved_settings = model_saved_settings or {}
#    train_settings.update(model_saved_settings)
#    model_weight_prefix = model_graph.get_weight_file_prefix()
#    training_thread = ThreadedModelTrainer(model_graph=model_graph,
#                                           # TODO initial weight is not needed
#                                           # initial_weights=model_weight_prefix,
#                                           train_settings=train_settings)
#    training_thread.start()
#    supervisor = training_thread.training_supervisor
#    thr = threading.Thread(target=send_email_every_minute)
#    thr.start()
#    app.run(host='0.0.0.0', port=5000)
