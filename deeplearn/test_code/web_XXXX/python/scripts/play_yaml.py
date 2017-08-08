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
import socket

from web.python.scripts.train_yaml import *

@app.route('/json/l_tests.json')
def list_tests_XX(min_date=None, max_date=None):
    test_list = model_graph.get_tests()
    list_ = []
    for file_ in test_list:
        x = json.loads(open(file_).read())
        list_.append(x)
    return json.dumps(list_)


@app.route('/ui/play')
def display_testing_status():
    template = 'nn_play.html'
    model_types = [] #bootstrap.list_model_types()
    return render_template(template,
                           host=socket.gethostname(),
                           #initial_state=get_training_status_struct(),
                           supervisor=model_graph,
                           model_types=model_types)



if __name__ == '__main__':
#    fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/tests/test2.yaml')
    #fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/bigru_cms_user_3.yaml')
    fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/gru_cms_user_2.yaml')
    #fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/gru_conv_cms_user_2.yaml')
#    fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/tests/test2.yaml')

    # model_name = ['Model_Tweet2Vec_BiGRU_CMSDataset',
    #              'Model_Tweet2Vec_BiGRU_UserCMSDataset',
    #              'Model_Tweet2Vec_BiGRU_BuzzometerDatasetVader']
    #model_name = model_name[1]
    #model_type = 'Tweet2Vec_BiGRU'
    model_graph = CompiledModel.load_yaml(fil)
    model_graph.build()
    model_graph.initialize()
    #model_saved_settings = model_graph.yaml_train_settings
    #model_saved_settings = model_saved_settings or {}
    #train_settings.update(model_saved_settings)
    #model_weight_prefix = model_graph.get_weight_file_prefix()
    #training_thread = ThreadedModelTrainer(model_graph=model_graph,
    #                                       # TODO initial weight is not needed
    #                                       # initial_weights=model_weight_prefix,
    #                                       train_settings=train_settings)
    #training_thread.start()
    #supervisor = training_thread.training_supervisor
    #thr = threading.Thread(target=send_email_every_minute)
    #thr.start()
    app.run(host='0.0.0.0', port=5000)
