# this script trains a models
# prototype for the file to copy in model folder

from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
import jinja2
import json
import os
import re
import random
import time
from threading import Thread
import psutil
import glob
import datetime
import threading
import logging


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


from web.python import bootstrap
from web.python.bootstrap import PersistentGraph, list_model_types
from web.python.training import PersistentTrainingSupervisor, ThreadedModelTrainer
from notify.send_mail import EmailNotification

TEMPLATES_ROOT = os.path.join(os.path.expanduser('~'),
                              'python',
                              #'Python', 'DJ',
                              'deep-learning',
                              'deeplearn',
                              'twitter_sentiment', 'web')
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


@app.route('/json/latest_test.json')
def get_latest_test():
    files = supervisor._meta.get_confusion_matrices()
    if len(files) > 0:
        f = files[0]
        with open(f) as matrix_file:
            x = matrix_file.read()
            return x
    return json.dumps(None)


@app.route('/json/training_progress.json')
def get_training_progress():
    return json.dumps({'batch_number': supervisor.batch_number,
                       'batches_per_epoch': supervisor.batches_per_epoch,
                       'epoch_number': supervisor.epoch_number,
                       'percent_epoch_complete': supervisor.epoch_percent,
                       'percent_training_complete': supervisor.training_percent,
                       'total_epochs': supervisor.number_of_epochs,
                       'batch_time': supervisor.batch_time.total_seconds(),
                       'epoch_time': supervisor.epoch_time.total_seconds(),
                       'elapsed_time': supervisor.elapsed_time.total_seconds(),
                       'remaining_time': supervisor.remaining_time.total_seconds()})


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
    return json.dumps({'loss': loss,
                       'accuracy': accuracy})

#    complete_path = os.path.join(root_dir(), path)
#    ext = os.path.splitext(path)[1]
#    mimetype = mimetypes.get(ext, "text/html")
#    content = get_file(complete_path)
#    return Response(content, mimetype=mimetype)
#


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


last_email_time = 0


@app.route('/site/report_testing_email.html')
def get_report_email():
    global last_email_time

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
    return render_template('email.html',
                           test_matrices=test_matrices,
                           supervisor=supervisor)


@app.route('/ui/training')
def display_training_status():
    template = 'nn_training.html'
    model_types = bootstrap.list_model_types()
    return render_template(template,
                           supervisor=supervisor,
                           model_types=model_types)


@app.route('/action/stop_training')
def stop_training():
    training_thread.stop()
    print('TRAINING STOPPED')
    return json.dumps({'status': 'ok'})


@app.route('/action/start_training')
def start_training():
    training_thread.start()
    print('TRAINING STARTED')
    return json.dumps({'status': 'ok'})


def send_email_every_minute():
    while True:
        print('Sending email')
        try:
            EmailNotification.sendEmail(get_report_email(), subject="Training report")
            time.sleep(3600)
        except:
            time.sleep(6)


supervisor = None

train_settings = {
    'optimizer': {
        'name': 'adam',
        'learning_rate': 0.001,
        'optimizer_parameters': {
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 0.00000001
        }
    },
    'validation_interval': 5,
    'test_interval': 1 * 60,
    'e_mail_interval': 1.5 * 3600,
    'summary_span': None,
    'checkpoint_interval': 450 * 60,
    'batch_size': 100,
    'validation_size': 100,
    'test_size': 1000,
    'epochs': 50
}

if __name__ == '__main__':
    model_name = ['Model_Tweet2Vec_BiGRU_CMSDataset',
                  'Model_Tweet2Vec_BiGRU_UserCMSDataset',
                  'Model_Tweet2Vec_BiGRU_BuzzometerDatasetVader']
    model_name = model_name[1]
    model_type = 'Tweet2Vec_BiGRU'
    model_graph = PersistentGraph.load(name=model_name, type_=model_type)
    model_saved_settings = model_graph.load_train_settings()
    model_saved_settings = model_saved_settings or {}
    train_settings.update(model_saved_settings)
    model_weight_prefix = model_graph.get_weight_file_prefix()
    training_thread = ThreadedModelTrainer(model_graph=model_graph,
                                           train_settings=train_settings)
    training_thread.start()
    supervisor = training_thread.training_supervisor
    thr = threading.Thread(target=send_email_every_minute)
    thr.start()
    app.run(host='0.0.0.0')
