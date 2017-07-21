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

@app.route('/ui')
def get_front_page():
    return render_template('nn_ui.html')

if __name__ == '__main__':
    app.run() #host='0.0.0.0')
