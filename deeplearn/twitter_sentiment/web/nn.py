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

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

loader = jinja2.FileSystemLoader(
    [os.path.join(os.path.dirname(__file__), "static"),
     os.path.join(os.path.dirname(__file__), "templates")])
environment = jinja2.Environment(loader=loader)


def render_template(template_filename, **context):
    return environment.get_template(template_filename).render(context)

@app.route('/site/static/<string:page_folder>/<string:page_name>')
def get_static_page(page_folder, page_name):
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


@app.route('/resource', methods = ['POST'])
def update_text():
    print(request.headers.get('contentType'))
    print("RES", request.get_json(force=True))
    return Response(json.dumps({"status":'ok'}), mimetype='application/json')


@app.route('/ui')
def get_front_page():
    page_template = request.args.get('page', None)
    template = 'nn_ui.html'
    if page_template is not None:
        template = "nn_"+page_template+'.html'
    server_data = {
        'model_types': bootstrap.list_model_types()
    }

    return render_template(template, model_types=bootstrap.list_model_types(),
                           server_data=server_data)

if __name__ == '__main__':
    app.run() #host='0.0.0.0')
