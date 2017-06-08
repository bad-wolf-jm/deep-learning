import os
import re
import time
import mimetypes

from functools import partial
from threading import Thread
from os.path import getsize
from datetime import datetime

from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout

from kivy.factory import Factory

kv_string = """
<GameHeader>:
    orientation: 'horizontal'
    title_color: .2,.2,.2,1
    main_color:  .1,.1,.1,1
    canvas:
        Color:
            rgba: 0.6,0.6,0.6,1
        Rectangle:
            size:self.size
            pos:self.pos

    BoxLayout:
        orientation: 'vertical'
        size_hint: 1, 1
        padding: [10,0,10,8]
        Label:
            text: 'SCORE'
            halign: 'left'
            valign:"middle"
            color: root.title_color
            font_size: 20
            size_hint: 1, 1
            width: self.texture_size[0]
            text_size: self.size
        Label:
            id:score_label
            color: root.main_color
            text: str(root.score)
            halign: 'left'
            valign: "middle"
            font_size: 45
            size_hint: 1, 1
            bold: True
            size: self.texture_size
            text_size: self.size

    Label:
        id:level_label
        color:   root.main_color
        text: root.level_number
        halign: 'center'
        valign: "middle"
        font_size: 45
        size_hint: 1, 1
        bold: True
        size: self.texture_size
        text_size: self.size

    BoxLayout:
        orientation: 'vertical'
        size_hint: 1, 1
        height: 30 #time_label.height+date_label.height
        padding: [10,0,10,8]
        Label:
            #id: time_label
            text: 'REMAINING'
            halign: 'right'
            valign:"middle"
            color: root.title_color
            font_size: 20
            size_hint: 1, 1
            width: self.texture_size[0]
            text_size: self.size
        Label:
            id:remaining_label
            color: root.main_color
            text: str(root.remaining_lives)
            halign: 'right'
            valign: "middle"
            font_size: 45
            size_hint: 1, 1
            bold: True
            size: self.texture_size
            text_size: self.size
"""

#from pydjay.core.keyboard import key_map

class GameHeader(BoxLayout):
    level_number    = StringProperty("LEVEL 1")
    score           = NumericProperty(0)
    remaining_lives = NumericProperty(5)

    def __init__(self, *args, **kwargs):
        super(GameHeader, self).__init__(*args, **kwargs)

Builder.load_string(kv_string)
Factory.register('GameHeader', GameHeader)
