from flask import Flask, Response
from flask import request  # , render_template
from flask_cors import CORS, cross_origin
import jinja2
import json
import os
import re
import random
from threading import Thread
import psutil
import glob

from models.rnn_classifier import train


import datetime

from notify.send_mail import EmailNotification

import time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

loader = jinja2.FileSystemLoader(
    [os.path.join(os.path.dirname(__file__), "static"),
     os.path.join(os.path.dirname(__file__), "templates")])
environment = jinja2.Environment(loader=loader)


def render_template(template_filename, **context):
    return environment.get_template(template_filename).render(context)


@app.route("/")
def hello():
    return json.dumps("Hello World!")


@app.route('/json/system_stats.json')
def get_system_info():
    memory = psutil.virtual_memory()
    used = memory.total - memory.available
    return json.dumps({'cpu': psutil.cpu_percent(),
                       'memory': [used, memory.total]})


@app.route('/json/training_stats.json')
def get_training_info():
    return json.dumps({'training': {'loss': train.supervisor.get_average_training_loss(15),
                                    'accuracy': train.supervisor.get_average_training_accuracy(15)},
                       'validation': {'loss': train.supervisor.get_average_validation_loss(15),
                                      'accuracy': train.supervisor.get_average_validation_accuracy(15)}})


@app.route('/json/training_progress.json')
def get_training_progress():
    return json.dumps({'batch_number': train.supervisor.batch_number,
                       'batches_per_epoch': train.supervisor.batches_per_epoch,
                       'epoch_number': train.supervisor.epoch_number,
                       'percent_epoch_complete': train.supervisor.epoch_percent,
                       'percent_training_complete': train.supervisor.training_percent,
                       'total_epochs': train.supervisor.number_of_epochs,
                       'batch_time': train.supervisor.batch_time.total_seconds(),
                       'epoch_time': train.supervisor.epoch_time.total_seconds(),
                       'elapsed_time': train.supervisor.elapsed_time.total_seconds(),
                       'remaining_time': train.supervisor.remaining_time.total_seconds()})


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

    loss = train.supervisor.get_loss_summary(min_timestamp, max_timestamp)
    accuracy = train.supervisor.get_accuracy_summary(min_timestamp, max_timestamp)
    return json.dumps({'loss': loss,
                       'accuracy': accuracy})

    complete_path = os.path.join(root_dir(), path)
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)


@app.route('/site/static/<string:page_folder>/<string:page_name>')
def get_page(page_folder, page_name):
    dir_name = os.path.dirname(__file__)
    page_path = os.path.join(dir_name, 'static', page_folder, page_name)
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }

    ext = os.path.splitext(page_name)[1]
    mimetype = mimetypes.get(ext, "text/html")

    print(page_path)
    return Response(open(page_path).read(), mimetype=mimetype)


@app.route('/site/report.json')
def get_report_page():
    foo = train.supervisor.get_test_results()
    # print(foo)
    return open(foo[0]).read()


@app.route('/site/test_results.html')
def get_test_result_page_html():
    foo = train.supervisor.get_test_results()

    #foo = glob.glob('/home/jalbert/.sentiment_analysis/training/ByteCNN/test/*.json')

    foo_data = sorted([[x, os.stat(x)] for x in foo], key=lambda x: x[1])
    list_ = []
    for file_ in foo_data:
        file_name = os.path.basename(file_[0])
        print(file_name)
        stats = os.stat(file_[0])
        #file_name, _ = os.path.splitext(file_name)
        regex = re.compile(r"test-(?P<test_number>\d+)-loss:(?P<loss>\d+(\.\d*)?)-accuracy:(?P<acc>\d+(\.\d*)?).json")
        metadata = regex.search(file_name)
        number = int(metadata.group('test_number'))
        loss = float(metadata.group('loss'))
        accuracy = float(metadata.group('acc'))
        list_.append({'path': file_,
                      'name': file_name,
                      'loss': loss,
                      'accuracy': accuracy,
                      'time': stats.st_mtime})
    return render_template('test_result_list.html', result_list=list_)

# if __name__ == '__main__':
#    get_test_result_page_html()
#    sys.exit(0)
#    foo = train.supervisor.get_test_results()
#    training_progress = json.loads(get_training_progress())
#    epoch_percent = '{:.2f}'.format(training_progress['percent_epoch_complete'])
#    remaining_time = datetime.datetime.utcfromtimestamp(training_progress['remaining_time']).strftime('%H:%M:%S')
#    elapsed_time = datetime.datetime.utcfromtimestamp(training_progress['elapsed_time']).strftime('%H:%M:%S')
#    training_progress['epoch_progress'] = '{:.2f}'.format(training_progress['percent_epoch_complete'])
#    return render_template('test_template.html',
#                           epoch_number=training_progress['epoch_number'],
#                           total_epochs=training_progress['total_epochs'],
#                           epoch_percent=epoch_percent,
#                           remaining_time=remaining_time,
#                           elapsed_time=elapsed_time)


def format_result_table(list_of_dicts):
    return render_template('test_result.html', table_rows=list_of_dicts)


#@app.route('/site/report_training_results.html')
# def get_report_result_page_html():
#    foo = train.supervisor.get_test_results()
#    bar = json.loads(open(foo[0]).read())
#    return format_result_table(bar['train']['output'])


#@app.route('/site/report_testing_results.html')
# def get_report_result_test_page_html():
#    foo = train.supervisor.get_test_results()
#    bar = json.loads(open(foo[0]).read())
#    return format_result_table(bar['test']['output'])


def format_confusion_matrix(labels, true_labels, predicted_labels):
    matrix = {}
    for i in labels:
        for j in labels:
            matrix[i, j] = 0
    for t_l, p_l in zip(true_labels, predicted_labels):
        if (t_l, p_l) not in matrix:
            matrix[(t_l, p_l)] = 0
        matrix[(t_l, p_l)] += 1
    return matrix  # render_template('confusion.html', matrix_entries=matrix)


#@app.route('/site/report_testing_results_confusion.html')
# def get_report_confusion_test_page_html():
#    foo = train.supervisor.get_test_results()
#    bar = json.loads(open(foo[0]).read())
#    true_values = [x['truth'] for x in bar['test']['output']]
#    predicted_values = [x['predicted'] for x in bar['test']['output']]
#    return format_confusion_matrix([0, 1, 2], true_values, predicted_values)


@app.route('/site/report_testing_email.html')
def get_report_email():
    foo = train.supervisor.get_test_results()
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
                           name=type(train.supervisor.model).__name__,
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


def send_email_every_minute():
    while True:
        print('Sending email')
        try:
            EmailNotification.sendEmail(get_report_email(), subject="Training report")
            time.sleep(3600)
        except:
            time.sleep(6)


if __name__ == '__main__':
    thr = Thread(target=train.start_training)
    thr.start()
    #thr2 = Thread(target=send_email_every_minute)
    # thr2.start()
    # print(train.foo)
    print('DDDDDDD')
    app.run()  # debug=True)
