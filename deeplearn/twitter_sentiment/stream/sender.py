import zmq
#import json
import threading
import time


class DataStreamer(threading.Thread):
    """docstring for RPCServer."""

    def __init__(self, name=None, host='127.0.0.1', port=9999, **kw):
        threading.Thread.__init__(self)
        self._name = name
        self._port = port
        self._host = host
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://{host}:{port}".format(host=self._host, port=self._port))
        #print('connected')

    def send(self, data_packet):
        try:
            sent = False
            while not sent:
                try:
                    self._socket.send_json(data_packet, flags=zmq.NOBLOCK)
                    sent = True
                except zmq.error.Again:
                    time.sleep(0.05)
            bar = self._socket.recv_json()
            return bar
        except Exception as e:
            print('ERROR', e)
            raise

    def disconnect(self):
        #print('disconnecting socket')
        self._socket.close()

    def shutdown(self):
        self.disconnect()
