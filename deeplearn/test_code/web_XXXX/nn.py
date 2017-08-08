from flask import Flask, Response
from flask import request  # , render_template
from flask_cors import CORS, cross_origin
import jinja2
import json
import os
import random
from threading import Thread
import psutil
import glob
import datetime
import time
import web.python.bootstrap as bootstrap
from web.python.bootstrap import PersistentGraph, list_model_types
from web.python.training import PersistentTrainingSupervisor, ThreadedModelTrainer
from notify.send_mail import EmailNotification


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

loader = jinja2.FileSystemLoader(
    [os.path.join(os.path.dirname(__file__), "static"),
     os.path.join(os.path.dirname(__file__), "templates")])
environment = jinja2.Environment(loader=loader)


def render_template(template_filename, **context):
    return environment.get_template(template_filename).render(context)


@app.route('/status/system_stats.json')
def get_system_info():
    memory = psutil.virtual_memory()
    used = memory.total - memory.available
    return json.dumps({'cpu': psutil.cpu_percent(),
                       'memory': [used, memory.total]})


@app.route('/site/static/<string:page_folder>/<string:page_name>')
def get_static_page(page_folder, page_name):
    print(page_name)
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


@app.route('/resource', methods=['POST'])
def update_text():
    print(request.headers.get('contentType'))
    print("RES", request.get_json(force=True))
    return Response(json.dumps({"status": 'ok'}), mimetype='application/json')


def get_page_template(template_name):
    templates = {
        'types': 'nn_front.html',
        'models': 'nn_model_type.html'
    }
    return templates.get(template_name, None)


@app.route('/ui/types')
def list_model_types():
    template = 'nn_front.html'
    model_types = bootstrap.list_model_types()
    return render_template(template, model_types=model_types)

@app.route('/ui/training')
def display_training_status():
    template = 'nn_training.html'
    model_types = bootstrap.list_model_types()
    return render_template(template, model_types=model_types)

@app.route('/ui/training_setup')
def display_training_setup():
    template = 'nn_training_setup.html'
    model_types = bootstrap.list_model_types()
    return render_template(template, model_types=model_types)

@app.route('/ui/testing_setup')
def display_testing_setup():
    template = 'nn_testing_setup.html'
    model_types = bootstrap.list_model_types()
    return render_template(template, model_types=model_types)


@app.route('/ui/models/<string:type_name>')
def list_type_instances(type_name):
    template = 'nn_model_type.html'
    type_data = bootstrap.get_type_specs(type_name)
    instance_list = bootstrap.list_type_instances(type_name)
    return render_template(template, model_type=type_data, instance_list=instance_list)


#@app.route('/ui/instance') # ?type=<type-name>&name=<instance-name>
@app.route('/actions/train_model')
def start_training():
    model_type = request.args.get('type', None)
    model_name = request.args.get('name', None)
    training_thread = ThreadedModelTrainer(model_name=model_name, model_type=model_type, train_settings=train_settings)
    training_thread.start()
    training_thread.ready_lock.acquire()
    supervisor = training_thread.training_supervisor
    template = 'nn_training.html'
    model_types = bootstrap.list_model_types()
    return render_template(template,
                           supervisor=supervisor,
                           model_types=model_types)

train_settings = {
    'optimizer': {
        'name': 'AdamOptimizer',
        'learning_rate': 0.01,
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
    'checkpoint_interval': 30 * 60,
    'batch_size': 100,
    'validation_size': 100,
    'test_size': 1000,
    'epochs': 50
}

#if __name__ == '__main__':
# main_training()
#lock.acquire()
#thread = threading.Thread(target=main_training)
#thread.start()
#lock.acquire()
#e_mail_thread = threading.Thread(target=send_email_every_minute)
#e_mail_thread.start()
training_thread = None #ThreadedModelTrainer(model_name='Model_Tweet2Vec_BiGRU_CMSDataset', model_type='Tweet2Vec_BiGRU', train_settings=train_settings)
#training_thread.start()
#training_thread.ready_lock.acquire()
supervisor = None #training_thread.training_supervisor
#app.run()



@app.route('/ui')
def get_front_page():
#    page_template = get_page_template(request.args.get('page', None))
#    model_type = request.args.get('type', None)
#    instance_name = request.args.get('name', None)
#
    template = 'nn_model_type.html'
#    if page_template is not None:
#        template = "nn_" + page_template + '.html'
    server_data = {
        'model_types': bootstrap.list_model_types()
    }
#
    return render_template(template,
                           model_types=bootstrap.list_model_instances(),
                           server_data=server_data)


if __name__ == '__main__':
    app.run()  # host='0.0.0.0')
