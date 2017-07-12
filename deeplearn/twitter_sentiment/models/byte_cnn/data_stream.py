# stream data
from stream.nn.streamer import TrainingDataStreamer
from models.byte_cnn.sentiment import generate_batches, flags, count_rows
#
#
#import zmq
import argparse
#import pymysql
import numpy as np
from config import db, stream


host, port = flags.stream_to.split(':')
port = int(port)
streamer = TrainingDataStreamer(validation_interval=flags.validation_interval, summary_span=None)

N = count_rows()
test = N // 100
batch_generator = generate_batches(min_id=test + 1, batch_size=flags.batch_size, epochs=flags.epochs)
validation_iterator = generate_batches(min_id=0, max_id=test, batch_size=flags.validation_size, epochs=None)
try:
    streamer.stream(batch_generator, validation_iterator, host=host, port=port)
except KeyboardInterrupt:
    streamer.streamer.shutdown()
