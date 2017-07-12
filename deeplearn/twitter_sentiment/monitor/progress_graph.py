""" Application's main screen."""
import time
import math
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
#from threading import Thread
import os
import numpy as np
#from kivy.garden import graph
#import numpy as np
import widgets
import clickable_area
#import math
#from ssh_connect import SSHConnect
#from stream.sender import DataStreamer
import collections
import datetime

kv_string = """
<TrainingProgressBox>:
    size_hint: 1,None
    height:55
    title: 'PROGRESS'
    epoch_progress: ""
    remaining_time: ""
    batch_index:0
    total_batches:1
    #is_percentage: False
    #graph:graph
    #stats: stats
    #graph: graph
    #validation_average: stats_display.validation_average
    Widget:
        canvas:
            Color:
                rgba: 0.3,0.3,0.3,1
            Rectangle:
                size:self.size
                pos:self.pos
        size_hint: 1,1
        pos_hint: {'x': 0, 'y': 0}

    BoxLayout:
        orientation: 'vertical'
        size_hint: 1,1

        BoxLayout:
            orientation: 'horizontal'
            size_hint: 1,1
            Label:
                bold: True
                halign: 'left'
                valign:"top"
                font_size: 20
                size_hint: 1, None
                height: 40
                width: self.texture_size[0]
                text_size: self.size
                text: root.epoch_progress #'Epoch 1 of 25 (34% complete)'
            BoxLayout:
                orientation: 'vertical'
                size_hint: 1,1
                Label:
                    bold: False
                    halign: 'right'
                    valign:"middle"
                    font_size: 15
                    size_hint: 1, None
                    height: 15
                    width: self.texture_size[0]
                    text_size: self.size
                    text: 'Remaining time:'
                Label:
                    bold: True
                    halign: 'right'
                    valign:"middle"
                    font_size: 20
                    size_hint: 1, None
                    height: 25
                    width: self.texture_size[0]
                    text_size: self.size
                    text: root.remaining_time #'15:23.255'

        ProgressBar:
            size_hint: 1, None
            height:15
            min:0
            max:root.total_batches
            value: root.batch_index
"""


class TrainingProgressBox(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super(TrainingProgressBox, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)
        self._remaining_time = None
        self._batch_times = []
        self._average_batch_time = None
        self._remaining_batches = 0
        self._force_change_remaining_time = 25

    def extend_graph(self, training=None, validation=None):
        if training is not None:
            self._training_data.extend(training)
            if len(training) > 0:
                self._min_displayed_batch_index = training[-1][0] + 1
        if validation is not None:
            self._validation_data.extend(validation)
        self.graph.extend_graph(training, validation)
        self.stats.extend_graph(training, validation)

    def _post_init(self, *args):
        pass

    def update(self, data):
        #print data.keys()
        #print data['batch_number'], data['batches_per_epoch']
        epoch_percent = float(data['batch_number']) / float(data['batches_per_epoch'])
        epoch_percent *= 100
        epoch_percent = int(math.floor(epoch_percent))
        epoch_progress = "Epoch {epoch_number} of {total_epochs} ({progress}% complete)"
        epoch_progress = epoch_progress.format(epoch_number=data['epoch_number'],
                                               total_epochs=data['total_epochs'],
                                               progress=epoch_percent)
        self._batch_times.extend(data['batch_time'])
        self._average_batch_time = np.mean([x[1] for x in self._batch_times[-50:]])
        self._remaining_batches = data['total_batches'] - data['batch_index']
        t = self._remaining_batches * self._average_batch_time
        _old_remaining_time = self._remaining_time
        self._remaining_time = min(self._remaining_time, t) if self._remaining_time is not None else t
        if self._remaining_time == _old_remaining_time:
            self._force_change_remaining_time -= 1
            print self._force_change_remaining_time
        else:
            print self._remaining_time, _old_remaining_time
            self._force_change_remaining_time = 25
        if self._force_change_remaining_time == 0:
            self._remaining_time = t
            self._force_change_remaining_time = 25
        self.epoch_progress = epoch_progress
        t = datetime.timedelta(seconds=self._remaining_time)
        self.remaining_time = str(t)
        #print self._average_batch_time, self._remaining_batches, self._remaining_time
        self.batch_index=data['batch_index']
        self.total_batches=data['total_batches']


Builder.load_string(kv_string)
Factory.register('TrainingProgressBox', TrainingProgressBox)


if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.clock import Clock
    from kivy.config import Config
    Config.set('kivy', 'exit_on_escape', '0')
    os.chdir('/Users/jihemme/python/DJ/deep-learning/deeplearn/twitter_sentiment')
    Window.clearcolor = (0,0,0, 1)
    Window.size = (864, 752)
    bar = TrainingProgressBox()
    try:
        runTouchApp(bar)
    except Exception, details:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print '-' * 60
        traceback.print_exc(file=sys.stdout)
        print '-' * 60
        print details
    finally:
        pass
