from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import json
import random
from threading import Thread

from models.byte_cnn import train


app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    return json.dumps("Hello World!")


@app.route('/json/system_stats.json')
def get_system_info():
    return json.dumps({'cpu': 36.25,
                       'memory': [11982673, 3493845]})


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


@app.route('/data/test')
def perform_test(series_name):
    num_samples = request.args.get('num_samples', None)
    return str(request.args)


if __name__ == '__main__':
    thr = Thread(target=train.start_training)
    thr.start()
    # print(train.foo)
    print('DDDDDDD')
    app.run()
