""" Application's main screen."""
import time
from kivy.factory import Factory
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from threading import Thread
import os
from kivy.garden import graph
import numpy as np
import widgets
import clickable_area
import math
from ssh_connect import SSHConnect
from stream.sender import DataStreamer
import collections

Factory.register('Graph', graph.Graph)

kv_string = """
<StatsDisplay@BoxLayout>:
    orientation: 'vertical'
    training_color: 1,0.2,0,1
    validation_color:0,1,0.5,1
    training_average: 'nan'
    validation_average: 'nan'
    is_percentage:False
    canvas:
        Color:
            rgba: 0,0,0,.85
        Rectangle:
            size:self.size
            pos:self.pos
    padding:[5,5,5,5]
    Label:
        size_hint: 1,None
        height: 19
        text: 'Averages'
        bold: True
        color: [1,1,1,1]
        halign: 'center'
        valign: 'middle'
        text_size: self.size

    BoxLayout:
        orientation: 'horizontal'
        size_hint: 1,1
        Label:
            size_hint: 1,1
            text: 'Training:'
            bold: False
            color: root.training_color
            halign: 'left'
            valign: 'middle'
            text_size: self.size
        Label:
            size_hint: 1,1
            text: root.training_average
            font_size:15
            bold: True
            color: root.training_color
            halign: 'right'
            valign: 'middle'
            text_size: self.size
    BoxLayout:
        orientation: 'horizontal'
        size_hint: 1,1
        Label:
            size_hint: 1,1
            text: 'Validation:'
            bold: False
            color: root.validation_color
            halign: 'left'
            valign: 'middle'
            text_size: self.size
        Label:
            size_hint: .5,1
            text: root.validation_average
            font_size:15
            bold: True
            color: root.validation_color
            halign: 'right'
            valign: 'middle'
            text_size: self.size
"""
class StatsDisplay(BoxLayout):
    def __init__(self, *args, **kwargs):
        super(StatsDisplay, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)
        self._training_data = collections.deque(maxlen=10)
        self._validation_data = collections.deque(maxlen=10)

    def extend_graph(self, training=None, validation=None):
        if training is not None:
            self._training_data.extend(training)
        if validation is not None:
            self._validation_data.extend(validation)
        self._update_average()

    @mainthread
    def _update_average(self):
        def __format(number):
            formats = {(0, 1): '{avg:.4f}',
                       (1, 10): '{avg:.3f}',
                       (10, 100): '{avg:.2f}',
                       (100, 1000): '{avg:.1f}',
                       (1000, None): '{avg:.0f}'}
            for interval in [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, None)]:
                l, u = interval
                if (u is None) or (number < u and number >= l):
                    return formats[interval].format(avg=number) + \
                        ("%" if self.is_percentage else "")

        try:
            training_avg = np.mean([x[1] for x in self._training_data])
            validation_avg = np.mean([x[1] for x in self._validation_data])
            self.training_average = __format(training_avg)
            self.validation_average = __format(validation_avg)
        except Exception, details:
            print details
            pass

    def _post_init(self, *args):
        pass
Builder.load_string(kv_string)
Factory.register('StatsDisplay', StatsDisplay)


kv_string = """
<TrainingStats>:
    is_percentage:False
    xlabel:''
    ylabel:''
    xmin:-0
    xmax:100
    ymin:0
    ymax:2
    x_ticks_minor:5
    x_ticks_major:25
    y_ticks_major:1
    y_ticks_minor:0.5
    y_grid_label:True
    x_grid_label:True
    padding:5
    x_grid:True
    y_grid:True
"""


class TrainingStats(graph.Graph):
    def __init__(self, *args, **kwargs):
        super(TrainingStats, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)
        self._training_data = []
        self._validation_data = []
        self._range = 250

    def extend_graph(self, training=None, validation=None):
        if training is not None:
            self._training_data.extend(training)
            if len(training) > 0:
                self._min_displayed_batch_index = training[-1][0] + 1
        if validation is not None:
            self._validation_data.extend(validation)
        self._update_graph_plots()

    @mainthread
    def _update_graph_plots(self):
        try:

            losses = self._training_data[-self._range:]
            validation_losses = self._validation_data[-self._range:]

            x_min = losses[0][0]
            x_max = max(losses[-1][0] + 10, self._range)
            x_amplitude = x_max - x_min
            x_max += (50 - x_amplitude % 50)
            self.xmin = x_min
            self.xmax = x_max
            self.x_ticks_major = (x_max - x_min) / 10
            self.x_ticks_minor = 5
            if self.is_percentage:
                self.ymin = 0
                self.ymax = 100
                self.y_ticks_major = 20
                self.y_ticks_minor = 5
            else:
                y_min = min([x[1] for x in self._training_data[-1 * self._range:]])
                y_min = math.floor(min(y_min * 1.1, 0))
                y_max = math.ceil(max([x[1] for x in self._training_data[-1 * self._range:]]) * 1.25)
                y_amplitude = y_max - y_min
                y_max += (10 - y_amplitude % 10)
                self.ymin = y_min
                self.ymax = y_max
                self.y_ticks_major = (y_max - y_min) / 5
                self.y_ticks_minor = 5
            self.training_plot.points = losses
            self.validation_plot.points = validation_losses
        except Exception, details:
            print details
            pass

    def _post_init(self, *args):
        self.training_plot = graph.SmoothLinePlot(color=[1, 0.2, 0, 1])
        self.validation_plot = graph.SmoothLinePlot(color=[0, 1, 0.5, 1])
        self.add_plot(self.validation_plot)
        self.add_plot(self.training_plot)

Builder.load_string(kv_string)
Factory.register('TrainingStats', TrainingStats)

kv_string = """
<TrainingStatsBox>:
    size_hint: 1,1
    title: 'STATS'
    is_percentage: False
    graph:graph
    stats: stats
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
        size_hint: 1, 1
        #height: 275
        canvas:
            Color:
                rgba: 0.5,0.5,0.5,1
            Rectangle:
                size:self.size
                pos:self.pos
        Label:
            canvas.before:
                Color:
                    rgba: 0.7,0.7,0.7,1
                Rectangle:
                    size:self.size
                    pos:self.pos

            id: time_label
            bold: True
            halign: 'center'
            valign:"middle"
            font_size: 15
            size_hint: 1, None
            height: 25
            width: self.texture_size[0]
            text_size: self.size
            text: root.title
        RelativeLayout:
            size_hint: 1,1
            TrainingStats:
                id: graph
                is_percentage:root.is_percentage
                size_hint: 1,1
                pos_hint: {'x':0, 'y':0}
                y_ticks_major:20
                y_ticks_minor:2
                ymin:0
                ymax:100
            StatsDisplay:
                id: stats
                is_percentage: root.is_percentage
                size_hint: None, None
                size: 175, 70
                pos_hint: {'right':1.0, 'top':1.0}
"""


class TrainingStatsBox(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super(TrainingStatsBox, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)
        self._training_data = []
        self._validation_data = []
        self._min_displayed_batch_index = 0
        self._range = 500
        self._first_data_fill = True

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
Factory.register('TrainingStatsBox', TrainingStatsBox)
