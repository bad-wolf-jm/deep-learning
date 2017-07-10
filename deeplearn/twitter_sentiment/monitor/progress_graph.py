""" Application's main screen."""
import time
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
#from threading import Thread
import os
#from kivy.garden import graph
#import numpy as np
import widgets
import clickable_area
#import math
#from ssh_connect import SSHConnect
#from stream.sender import DataStreamer
import collections

kv_string = """
<TrainingProgressBox>:
    size_hint: 1,None
    height:55
    title: 'PROGRESS'
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
                text: 'Epoch 1 of 25 (34% complete)'
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
                    text: '15:23.255'

        ProgressBar:
            size_hint: 1, None
            height:15
            min:0
            max:100
            value: 65
"""


class TrainingProgressBox(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super(TrainingProgressBox, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)
        #self._first_data_fill = True

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
