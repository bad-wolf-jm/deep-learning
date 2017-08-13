from flask import Flask, Response, request
from flask_cors import CORS
import jinja2
import json
import os
from threading import Lock
import psutil
import threading
import logging
import socket
import tensorflow as tf


log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)


D = os.path.dirname(__file__)
print(__file__)
print(D)

TEMPLATES_ROOT = D

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
    with latest_test_lock:
        _ = make_test_output_matrix(latest_test)
    return json.dumps(_)

#@app.route('/json/latest_test_output.json')
#def get_latest_test_output():
#    global latest_test
#    with latest_test_lock:
#        _ = latest_test.output
#        return json.dumps(_)



def get_training_status_struct():
    return {'batch_number': supervisor.batch_number if supervisor is not None else 0,
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


def get_float_arg(request, arg_name):
    try:
        return float(request.args.get(arg_name))
    except:
        return None


@app.route('/json/training_graphs.json')
def get_training_graph_series():
    min_timestamp = get_float_arg(request, 'min_timestamp')
    max_timestamp = get_float_arg(request, 'max_timestamp')
    loss = supervisor.get_loss_summary(min_timestamp, max_timestamp)
    accuracy = supervisor.get_accuracy_summary(min_timestamp, max_timestamp)
    return json.dumps({'loss': loss, 'accuracy': accuracy})


@app.route('/static/<string:page_folder>/<path:page_name>')
def get_page(page_folder, page_name):
    dir_name = TEMPLATES_ROOT
    page_path = os.path.join(dir_name, 'static', page_folder, page_name)
    print(page_path)
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript"
    }

    ext = os.path.splitext(page_name)[1]
    mimetype = mimetypes.get(ext, "application/octet-stream")
    try:
        return Response(open(page_path).read(), mimetype=mimetype)
    except Exception as e:
        print(e)
        return app.send_static_file(page_path)


@app.route('/ui/training')
def display_training_status():
    template = 'nn_training.html'
    model_types = []
    return render_template(template,
                           host=socket.gethostname(),
                           initial_state=get_training_status_struct(),
                           supervisor=supervisor,
                           model_types=model_types)


supervisor = None
latest_test = None
latest_test_lock = threading.Lock()


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
