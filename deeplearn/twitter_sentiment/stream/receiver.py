import zmq
#import json
import threading
import time


class DataReceiver(threading.Thread):
    """docstring for RPCServer."""

    def __init__(self, name=None, bind='127.0.0.1', port=9999, **kw):
        threading.Thread.__init__(self)
        self._name = name
        self._port = port
        self._host = bind
        self._running = False
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://{host}:{port}".format(host=self._host, port=self._port))

        self._action_handlers = {}

    def run(self):
        while self._running:
            try:
                x = self._socket.recv_json()
                action = x.get('action', None)
                #print (action)
                if action is not None:
                    callback = self._action_handlers.get(action, None)
                    if callback is not None:
                        try:
                            return_value = callback(**x.get('payload', None))
                            print(return_value)
                            self._socket.send_json({'status': 'ok', 'return': return_value})
                        except Exception as e:
                            print(e)
                            self._socket.send_json({'status': 'error', 'return': str(e)})
                    else:
                        self._socket.send_json({'status': 'ok', 'return': None})
                else:
                    self._socket.send_json({'status': 'ok', 'return': None})

            except Exception as details:
                print ("ERROR", details)
        self.__socket.close()

    def register_action_handler(self, action, fnc):
        self._action_handlers[action] = fnc

    def start(self, threaded=True):
        self._running = True
        if not threaded:
            self.run()
        else:
            threading.Thread.start(self)

    def stop(self):
        self._running = False
