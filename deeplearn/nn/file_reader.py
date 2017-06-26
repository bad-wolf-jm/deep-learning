# This file is part of audioread.
# Copyright 2011, Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

#from __future__ import with_statement
#from __future__ import division

#import sys
import threading
#import os
#import traceback
#import jack
#import time
import array
from multiprocessing import Process, Queue

#from decoder import get_loop_thread

try:
    import queue
except ImportError:
    import Queue as queue


QUEUE_SIZE = 100
BUFFER_SIZE = 100


class FileReaderProcess(Process):
    """
    Reads a file in a separate process. The contents of the file is sent to the process' output
    queue.
    uri:  file://path/to/file
          zip://path/to/zip/file//path/to/resources
    """
    def __init__(self, uri = None):
        super(FileReaderProcess, self).__init__()

        self._uri = uri
        try:
            protocol, path = self._uri.split('://')
        except:
            protocol = 'file'
            path = self._uri

        try:
            opener = getattr(self, 'open_'+protocol)
            self._file
        except AttributeError:
            print("I don't know how to open this")

        self.in_queue = in_queue
        self.out_queue = out_queue
        self.client_name = name
        self.num_channels = num_channels
        self.player = None

    def run(self):
        self.output_driver = JackOutputDriver(self.client_name, self.num_channels)
        self.out_queue.put(('_init', (), {"client_name":  self.output_driver.client_name,
                                          "block_size":   self.output_driver.block_size,
                                          "samplerate":   self.output_driver.samplerate,
                                          "num_channels": self.output_driver.num_channels}))
        while True:
            try:
                command, args, kwargs = self.in_queue.get_nowait()
                if command == 'close':
                    print "CLOSING:", self.client_name
                    break
                try:
                    getattr(self.output_driver, command)(*args, **kwargs)
                except AttributeError, details:
                    pass
                except Exception, details:
                    print 'y', details
            except queue.Empty, details:
                pass
            self.out_queue.put(
                ('set_stream_time', (self.output_driver.stream_time, self.output_driver.buffer_time), {}))


class JackOutput(object):
    def __init__(self, client_name="PYDjayJackClient", num_channels=2, *args, **kw):
        super(JackOutput, self).__init__()
        self.out_queue = Queue(maxsize=10)
        self.in_queue = Queue(maxsize=100)
        self.ready_sem = threading.Semaphore(0)
        self._output_process = JackOutputProcess(client_name, num_channels,
                                                 self.out_queue, self.in_queue)
        self._output_process.daemon = True
        self._output_process.start()
        self._running = True
        self._foo = threading.Thread(target=self._print_info)
        self._foo.start()
        self.stream_time = 34
        self.ready_sem.acquire()

    def _init(self, block_size=0, samplerate=0, client_name="", num_channels=0):
        self.block_size = block_size
        self.samplerate = samplerate
        self.client_name = client_name
        self.num_channels = num_channels
        self.ready_sem.release()

    def _print_info(self):
        while self._running:
            try:
                command, args, kwargs = self.in_queue.get_nowait()
                if command == 'QUIT':
                    break
                try:
                    getattr(self, command)(*args, **kwargs)
                    # time.sleep(.1)
                except AttributeError, details:
                    print details
                    pass
                except Exception, details:
                    print 'y', details
            except queue.Empty, details:
                pass
            finally:
                pass

    def connect_outputs(self, **kwargs):
        self.out_queue.put(('connect_outputs', (), kwargs))

    def disconnect_outputs(self, **kwargs):
        self.out_queue.put(('disconnect_outputs', (), kwargs))

    def flush_buffer(self):
        self.out_queue.put(('flush_buffer', (), {}))

    def reset_timer(self, timestamp=0):
        self.out_queue.put(('reset_timer', (), {'timestamp': timestamp}))

    def set_stream_time(self, time, b_time):
        self.stream_time = time
        self.buffer_time = b_time

    def send(self, data):
        # print 'sending'
        self.out_queue.put(('send', (data,), {}))

    def close(self):
        self.out_queue.put(('close', (), {}))
        self.out_queue.cancel_join_thread()
        self.in_queue.cancel_join_thread()
        self._running = False
        self._foo.join()
        self.out_queue.close()
        self.in_queue.close()
