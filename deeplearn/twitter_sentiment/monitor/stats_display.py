""" Application's main screen."""
import time
#from kivy.factory import Factory
from kivy.clock import Clock
from kivy.lang import Builder
#from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from threading import Thread
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
#from ssh_connect import SSHConnect
from stream.sender import DataStreamer
import stats_graph
import progress_graph
#import pydjay.bootstrap
#from pydjay.core.keyboard import key_map
#Factory.register('Graph', graph.Graph)


kv_string = """
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

        Label:
            id: time_label
            bold: True
            halign: 'center'
            valign:"middle"
            font_size: 35
            size_hint: 1, None
            height: 40
            width: self.texture_size[0]
            text_size: self.size
            text: "TRAINING MODEL"
        BoxLayout:
            orientation: 'vertical'
            size_hint: 1,1
            spacing: 10
            padding:[10,10,10,10]

            TrainingProgressBox:
                size_hint:1,None
                height:55

            TrainingStatsBox:
                id: loss_graph
                title: "LOSS"
                size_hint:1,1

            TrainingStatsBox:
                id: accuracy_graph
                title: "ACCURACY"
                is_percentage: True
                size_hint:1,1
"""

class MainScreen(FloatLayout):
    def __init__(self, *args, **kwargs):
        super(MainScreen, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)
        self._min_displayed_batch_index = 0
        self._range = 500
        self._first_data_fill = True

    def start_monitor(self):
        self._monitor_training = True
        self._monitor_thread = Thread(target=self._training_data_stream)
        self._monitor_thread.start()

    def stop_monitor(self):
        self._monitor_training = False
        self._monitor_thread.join()

    def _training_data_stream(self):
        self._graph_feed = DataStreamer(port=99887)
        while self._monitor_training:
            self._update_graph()
            time.sleep(1)

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
        #print (train_data)
        validation_data = validation_data.get('return', None)
        self.loss_graph.extend_graph(train_data['loss'], validation_data['loss'])
        self.accuracy_graph.extend_graph([[i, 100*j] for i,j in train_data['accuracy']], [[i, 100*j] for i,j in validation_data['accuracy']])

        if len(train_data['accuracy']) > 0:
            self._min_displayed_batch_index = train_data['accuracy'][-1][0] + 1

    def _post_init(self, *args):
        self.start_monitor()

Builder.load_string(kv_string)
