""" Application's main screen."""
#import os
#if __name__ == '__main__':
#    from kivy.base import runTouchApp
#    from kivy.core.window import Window
#    from kivy.clock import Clock
#    from kivy.config import Config
#    Config.set('kivy', 'exit_on_escape', '0')
#    os.chdir('/Users/jihemme/python/DJ/deep-learning/deeplearn/twitter_sentiment')


import time
from kivy.factory import Factory
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
#from threading import Thread
#import main_queue
#import player_display
#import player_deck\
import os
#print os.getcwd()
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
#from stream.sender import DataStreamer
#import stats_graph
#import pydjay.bootstrap
#from pydjay.core.keyboard import key_map
#Factory.register('Graph', graph.Graph)


kv_string = """
<Cell@Label>:
    size_hint: 1, None
    height:75
    canvas.before:
        Color:
            rgba: 0.5,0.5,0.5,1
        Rectangle:
            size:self.size
            pos:self.pos
<ColumnHeaderCell@Cell>:
    size_hint: 1, None
    height:75
    color:1,1,1,1
    bold:True
    canvas.before:
        Color:
            rgba: 0.2,0.2,0.2,1
        Rectangle:
            size:self.size
            pos:self.pos

<RowHeaderCell@Cell>:
    size_hint: 1, None
    height:75
    color: 1,1,1,1
    bold:True
    canvas.before:
        Color:
            rgba: 0.2,0.2,0.2,1
        Rectangle:
            size:self.size
            pos:self.pos

<ConfusionMatrix>:
    size_hint: 1,1
    #accuracy_graph: accuracy_graph
    #loss_graph: loss_graph

    Widget:
        canvas:
            Color:
                rgba: 0.3,0.3,0.3,1
            Rectangle:
                size:self.size
                pos:self.pos
        size_hint: 1,1
        pos_hint: {'x': 0, 'y': 0}

    GridLayout:
        cols: 4
        spacing: 1
        size_hint: 1,1
        pos_hint: {'x': 0, 'y': 0}
        ColumnHeaderCell:
            text: ''
        ColumnHeaderCell:
            text: 'POSITIVE'
        ColumnHeaderCell:
            text: 'NEGATIVE'
        ColumnHeaderCell:
            text: 'NEUTRAL'
        RowHeaderCell:
            text: 'POSITIVE'
        Cell:
            id: cell_1_1
            text: '1'
        Cell:
            id: cell_1_2
            text: '1'
        Cell:
            id: cell_1_3
            text: '1'
        RowHeaderCell:
            text: 'NEGATIVE'
        Cell:
            id: cell_2_1
            text: '1'
        Cell:
            id: cell_2_2
            text: '1'
        Cell:
            id: cell_2_3
            text: '1'
        RowHeaderCell:
            text: 'NEUTRAL'
        Cell:
            id: cell_3_1
            text: '1'
        Cell:
            id: cell_3_2
            text:'1'
        Cell:
            id: cell_3_3
            text: '1'
"""

class ConfusionMatrix(FloatLayout):
    def __init__(self, *args, **kwargs):
        super(ConfusionMatrix, self).__init__(*args, **kwargs)
        Clock.schedule_once(self._post_init, -1)

    def _post_init(self, *args):
        pass

Builder.load_string(kv_string)


if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.clock import Clock
    from kivy.config import Config
    Config.set('kivy', 'exit_on_escape', '0')
    os.chdir('/Users/jihemme/python/DJ/deep-learning/deeplearn/twitter_sentiment')
    Window.clearcolor = (0,0,0, 1)
    Window.size = (864, 752)
    bar = ConfusionMatrix()
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
