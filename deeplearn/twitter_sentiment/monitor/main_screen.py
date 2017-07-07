""" Application's main screen."""
import time
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
#import main_queue
#import player_display
#import player_deck\
import os
print os.getcwd()
#from main_window import MainWindow
#from track_editor import TrackEditor
#from dialogs.playlist_view import PlaylistView
#from dialogs.playlist_selector import PlaylistSelector
#from preview_player import PreviewPlayer
#from mixer import Mixer
from kivy.garden import graph
#from main_track_list import MainTrackList
#from track_short_list import TrackShortList
#from kivy.uix.treeview import TreeViewLabel
import widgets
import clickable_area
from ssh_connect import SSHConnect
from stream.sender import DataStreamer
#import pydjay.bootstrap
#from pydjay.core.keyboard import key_map
Factory.register('Graph', graph.Graph)
kv_string = """
<LabelledTextInput@BoxLayout>:
    label: "<default>"
    password: False
    label_halign:'left'
    label_valign:'middle'
    label_width: 50
    label_height: 20
    label_color:1,1,1,1
    bold:False
    spacing: 3

    Label:
        size_hint: None if root.orientation == 'horizontal' else 1, 1 if root.orientation == 'horizontal' else None
        height: root.height if root.orientation == 'horizontal' else root.label_height
        width: root.label_width
        text: root.label
        bold: root.bold
        color: root.label_color
        halign: root.label_halign
        valign: root.label_valign
        text_size: self.size

    TextInput:
        size_hint: 1, 1
        password: root.password

<StatsDisplay@BoxLayout>:
    orientation : 'vertical'
    training_color:1,0.2,0,1
    validation_color:0,1,0.5,1
    avg_loss: '4.7523'
    avg_accuracy: '100.0%'
    canvas:
        Color:
            rgba: 0,0,0,.5
        Rectangle:
            size:self.size
            pos:self.pos
    padding:[5,5,5,5]
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
            text: root.avg_loss
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
            text: root.avg_accuracy
            bold: True
            color: root.validation_color
            halign: 'right'
            valign: 'middle'
            text_size: self.size


<TrainingStats@Graph>:
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

<ConnectionStatus@BoxLayout>:
    orientation: 'horizontal'
    size_hint: 1, None
    height: 45
    BoxLayout:
        orientation: 'vertical'
        size_hint: 1,1
        Label:
            size_hint: 1,1
            halign:'left'
            valign:'middle'
            text_size: self.size
            multiline:True
            markup:True
            font_size:13
            text: "Program will run on 'localhost'"
        Label:
            size_hint: 1,1
            halign:'left'
            valign:'middle'
            text_size: self.size
            multiline:True
            markup:True
            font_size:13
            text: "Root: '~/path/to/cwd'"
        #Label:
        #    size_hint: 1,1
        #    halign:'left'
        #    valign:'middle'
        #    text_size: self.size
        #    multiline:True
        #    markup:True
        #    font_size:12
        #    text: "Virtualenv: 'None'"


    ImageButton:
        size_hint: None, 1
        width: self.height
        image_width: self.width - 10
        image_height: self.height - 10
        image:'monitor/icon-airplay.png'

<MainScreen>:
    size_hint: 1,1
    accuracy_graph: accuracy_graph
    loss_graph: loss_graph

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
            size_hint: 1, None
            height: 50 #time_label.height+date_label.height

            Label:
                id: time_label
                bold: True
                halign: 'center'
                valign:"middle"
                font_size: 35
                size_hint: 1, 1
                width: self.texture_size[0]
                text_size: self.size
                text: "TRAINING MODEL"
        BoxLayout:
            orientation: 'vertical'
            size_hint: 1,1
            spacing: 10
            padding:[10,10,10,10]

            BoxLayout:
                orientation: 'horizontal'
                size_hint: 1,None
                height: 300
                spacing: 10
                BoxLayout:
                    orientation: 'vertical'
                    size_hint: 1, 1
                    #height: 250
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
                        font_size: 20
                        size_hint: 1, None
                        height: 35
                        width: self.texture_size[0]
                        #size: self.texture_size
                        text_size: self.size
                        text: "TRAINING DATA FEED"
                    BoxLayout:
                        size_hint: 1, 1
                        orientation: 'vertical'
                        padding:[10,10,10,10]
                        spacing: 7
                        ConnectionStatus:
                        Widget:
                            size_hint: None, None
                            height:10
                        LabelledTextInput:
                            orientation: 'horizontal'
                            size_hint: 1, None
                            height: 30
                            label_width:40
                            label: 'Feed:'

                        BoxLayout:
                            orientation: 'horizontal'
                            size_hint: 1, None
                            height: 30
                            spacing: 20
                            LabelledTextInput:
                                orientation: 'horizontal'
                                size_hint: 1, None
                                height: 30
                                label_width:55
                                label: "Epochs:"
                            LabelledTextInput:
                                orientation: 'horizontal'
                                size_hint: 1, None
                                height: 30
                                label_width:75
                                label: "Batch size:"

                        BoxLayout:
                            orientation: 'vertical'
                            size_hint: 1, 1
                            height: 80
                            Label:
                                size_hint: 1, None
                                height: 30
                                bold: False
                                halign: 'left'
                                valign:"middle"
                                text_size: self.size
                                text: 'Other Arguments:'
                            TextInput:
                                size_hint: 1, 1

                BoxLayout:
                    orientation: 'vertical'
                    size_hint: 1,1
                    #height:
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
                        font_size: 20
                        size_hint: 1, None
                        height: 35
                        width: self.texture_size[0]
                        text_size: self.size
                        text: "TRAINING SERVER"
                    BoxLayout:
                        size_hint: 1, 1
                        orientation: 'vertical'
                        padding:[10,10,10,10]
                        spacing: 7

                        ConnectionStatus:
                        Widget:
                            size_hint: None, None
                            height:10

                        BoxLayout:
                            orientation: 'horizontal'
                            size_hint: 1, None
                            spacing: 3
                            height: 30
                            Label:
                                size_hint: None,1
                                height: 20
                                width:40
                                text: 'Feed:'
                                bold: False
                                halign: 'left'
                                valign: "middle"
                                text_size: self.size
                            TextInput:
                                size_hint: 1, 1

                        BoxLayout:
                            orientation: 'horizontal'
                            size_hint: 1, None
                            height: 30
                            spacing: 20
                            BoxLayout:
                                orientation: 'horizontal'
                                size_hint: 1, 1
                                height: 30
                                Label:
                                    size_hint: None, 1
                                    width: 75
                                    bold: False
                                    halign: 'left'
                                    valign:"middle"
                                    text_size: self.size
                                    text: 'Batch size:'
                                TextInput:
                                    size_hint: 1, 1
                        BoxLayout:
                            orientation: 'vertical'
                            size_hint: 1, 1
                            height: 80
                            Label:
                                size_hint: 1, None
                                height: 30
                                bold: False
                                halign: 'left'
                                valign:"middle"
                                text_size: self.size
                                text: 'Other Arguments:'
                            TextInput:
                                size_hint: 1, 1

            BoxLayout:
                orientation: 'vertical'
                size_hint: 1,1
                spacing: 10
                BoxLayout:
                    orientation: 'vertical'
                    size_hint: 1, None
                    height: 350
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
                        font_size: 20
                        size_hint: 1, None
                        height: 35
                        width: self.texture_size[0]
                        text_size: self.size
                        text: "LOSS"
                    RelativeLayout:
                        size_hint: 1,1
                        TrainingStats:
                            id: loss_graph
                            size_hint: 1,1
                            pos_hint: {'x':0, 'y':0}
                            y_ticks_major:20
                            y_ticks_minor:2
                            ymin:0
                            ymax:100
                        StatsDisplay:
                            size_hint: None, None
                            size: 175, 50
                            pos_hint: {'right':1.0, 'top':1.0}

                BoxLayout:
                    orientation: 'vertical'
                    size_hint: 1, None
                    height: 350
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
                        font_size: 20
                        size_hint: 1, None
                        height: 35
                        width: self.texture_size[0]
                        #size: self.texture_size
                        text_size: self.size
                        text: "ACCURACY"
                    RelativeLayout:
                        size_hint: 1,1
                        TrainingStats:
                            id: accuracy_graph
                            size_hint: 1,1
                            pos_hint: {'x':0, 'y':0}
                            ymin:0
                            ymax:100
                            y_ticks_major:10
                        StatsDisplay:
                            size_hint: None, None
                            size: 175, 50
                            pos_hint: {'right':1.0, 'top':1.0}

                Widget:
                    size_hint: 1,1
                Button:
                    text: 'Start training'
"""

from kivy.utils import get_color_from_hex as rgb


class MainScreen(FloatLayout):
    def __init__(self, *args, **kwargs):
        super(MainScreen, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)
        #self._graph_feed = DataStreamer(port=99887)
        #Clock.schedule_interval(self._update_graph, 1)
        self._training_loss_data = []
        self._training_accuracy_data = []
        self._validation_loss_data = []
        self._validation_accuracy_data = []
        self._min_displayed_batch_index = 0
        self._range = 500
        self._first_data_fill = True

        #self._max_displayed_batch_index = None

    def _update_graph(self, *a):
        if self._first_data_fill:
            train_data = self._graph_feed.send({'action': 'get_train_batch_data',
                                                'payload': {"min_batch_index": self._min_displayed_batch_index,
                                                            'max_batch_index': None}})
            validation_data = self._graph_feed.send({'action': 'get_validation_batch_data',
                                                     'payload': {"min_batch_index": self._min_displayed_batch_index,
                                                                 'max_batch_index': None}})
            self._first_data_fill = False
        else:
            train_data = self._graph_feed.send({'action': 'get_train_batch_data',
                                                'payload': {"min_batch_index": self._min_displayed_batch_index,
                                                            'max_batch_index': self._min_displayed_batch_index + self._range}})

            validation_data = self._graph_feed.send({'action': 'get_validation_batch_data',
                                                     'payload': {"min_batch_index": self._min_displayed_batch_index,
                                                                 'max_batch_index': None}})
        train_data = train_data.get('return', None)
        validation_data = validation_data.get('return', None)
        # print()
        # print()
        #print (train_data)
        #print (validation_data)
        if train_data is not None:
            self._training_loss_data.extend(train_data['losses'])
            self._training_accuracy_data.extend(train_data['accuracies'])
            self._validation_loss_data.extend(validation_data['losses'])
            self._validation_accuracy_data.extend(validation_data['accuracies'])
            if len(train_data['accuracies']) > 0:
                self._min_displayed_batch_index = train_data['accuracies'][-1][0] + 1
            self._update_plots()

    def _update_plots(self):
        try:
            losses = self._training_loss_data[-self._range:]
            accuracies = self._training_accuracy_data[-self._range:]
            validation_losses = self._validation_loss_data[-self._range:]
            validation_accuracies = self._validation_accuracy_data[-self._range:]
            self.loss_graph.xmin = losses[0][0]
            self.loss_graph.xmax = max(losses[-1][0] + 100, self._range)
            self.loss_graph.ymin = min([x[1] for x in self._training_loss_data[-4 * self._range:]])
            self.loss_graph.ymax = max([x[1] for x in self._training_loss_data[-4 * self._range:]]) * 1.5
            self.training_loss_plot.points = losses
            self.validation_loss_plot.points = validation_losses
            self.accuracy_graph.xmin = accuracies[0][0]
            self.accuracy_graph.xmax = max(accuracies[-1][0] + 100, self._range)
            self.training_accuracy_plot.points = [[i, 100 * x] for i, x in accuracies]
            self.validation_accuracy_plot.points = [[i, 100 * x] for i, x in validation_accuracies]
        except:
            pass

    def _post_init(self, *args):
        self.validation_loss_plot = graph.SmoothLinePlot(color=[0, 1, 0, 1])
        self.training_loss_plot = graph.SmoothLinePlot(color=[.5, .5, .9, 1])
        self.loss_graph.add_plot(self.validation_loss_plot)
        self.loss_graph.add_plot(self.training_loss_plot)

        self.validation_accuracy_plot = graph.SmoothLinePlot(color=[0, 1, 0, 1])
        self.training_accuracy_plot = graph.SmoothLinePlot(color=[.5, .5, .5, 1])
        self.accuracy_graph.add_plot(self.validation_accuracy_plot)
        self.accuracy_graph.add_plot(self.training_accuracy_plot)

        #foo = SSHConnect()
        # foo.open()


Builder.load_string(kv_string)
