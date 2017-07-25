# this script trains a models
# prototype for the file to copy in model folder

from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
import jinja2
import json
import os
import re
import random
from threading import Thread
import psutil
import glob
import datetime
import threading

from web.python import bootstrap
from web.python.bootstrap import PersistentGraph, list_model_types
from web.python.training import PersistentTrainingSupervisor


TEMPLATES_ROOT = os.path.join(os.path.expanduser('~'),
                              'python',
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

    complete_path = os.path.join(root_dir(), path)
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)


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


@app.route('/site/report_testing_email.html')
def get_report_email():
    foo = supervisor.get_test_results()
    bar = json.loads(open(foo[0]).read())
    training_progress = json.loads(get_training_progress())
    test_true_values = [x['truth'] for x in bar['test']['output']]
    test_predicted_values = [x['predicted'] for x in bar['test']['output']]
    test_confusion_matrix = format_confusion_matrix([0, 1, 2], test_true_values, test_predicted_values)

    train_true_values = [x['truth'] for x in bar['train']['output']]
    train_predicted_values = [x['predicted'] for x in bar['train']['output']]
    train_confusion_matrix = format_confusion_matrix([0, 1, 2], train_true_values, train_predicted_values)

    test_loss = bar['test']['loss']
    test_accuracy = bar['test']['accuracy'] * 100
    train_loss = bar['train']['loss']
    train_accuracy = bar['train']['accuracy'] * 100

    epoch_percent = '{:.2f}'.format(training_progress['percent_epoch_complete'])
    remaining_time = datetime.datetime.utcfromtimestamp(training_progress['remaining_time']).strftime('%H:%M:%S')
    elapsed_time = datetime.datetime.utcfromtimestamp(training_progress['elapsed_time']).strftime('%H:%M:%S')
    training_progress['epoch_progress'] = '{:.2f}'.format(training_progress['percent_epoch_complete'])

    return render_template('email.html',
                           name=type(supervisor.model).__name__,
                           epoch_number=training_progress['epoch_number'],
                           total_epochs=training_progress['total_epochs'],
                           epoch_percent=epoch_percent,
                           remaining_time=remaining_time,
                           elapsed_time=elapsed_time,
                           test_loss=test_loss,
                           test_accuracy=test_accuracy,
                           train_loss=test_loss,
                           train_accuracy=test_accuracy,
                           train_confusion_matrix=train_confusion_matrix,
                           test_confusion_matrix=test_confusion_matrix,
                           test_table_rows=bar['test']['output'],
                           train_table_rows=bar['train']['output'])


@app.route('/ui/training')
def display_training_status():
    template = 'nn_training.html'
    model_types = bootstrap.list_model_types()
    return render_template(template, model_types=model_types)

#def send_email_every_minute():
#    while True:
#        print('Sending email')
#        try:
#            EmailNotification.sendEmail(get_report_email(), subject="Training report")
#            time.sleep(3600)
#        except:
#            time.sleep(6)





supervisor = None


def main_training():
    global supervisor
    q = PersistentGraph.load(name="Script Create", type_="ByteCNN")
    q.initialize(session=None, training=True, resume=False)
    train_settings = q.load_train_settings()

    # NOTE Lookup the training data, like batch size, validation_size, test interval and number_of_classes
    # of epochs to pass them to the train_model function
    # - validation_interval
    # - validation_size
    # - test_interval
    # - test_size
    # - checkpoint_interval
    # - number of epochs
    # - e-mail interval

    train_settings = {
        'validation_interval': 5,
        'test_interval': 15 * 60,
        'e_mail_interval': 3600,
        'summary_span': None,
        'checkpoint_interval': 30 * 60,
        'batch_size': 100,
        'validation_size': 100,
        'test_size': 1000,
        'epochs': 10
    }

    model_saved_settings = q.load_train_settings()
    train_settings.update(model_saved_settings)

    supervisor = PersistentTrainingSupervisor(q,
                                              validation_interval=train_settings['validation_interval'],
                                              test_interval=train_settings['test_interval'],
                                              summary_span=train_settings['summary_span'],
                                              checkpoint_interval=train_settings['checkpoint_interval'])
    supervisor.train_model(batch_size=train_settings['batch_size'],
                           validation_size=train_settings['validation_size'],
                           test_size=train_settings['test_size'],
                           epochs=train_settings['epochs'])


if __name__ == '__main__':
    #q = PersistentGraph.load(name="Script Create", type_="ByteCNN")
    #q.initialize(session=None, training=True, resume=False)
    main_training()
    #thread = threading.Thread(target=main_training)
    #thread.start()
    #app.run()
