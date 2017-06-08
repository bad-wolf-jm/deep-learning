#from kivy.app import App
#from kivy.lang import Builder
from kivy.uix.widget import Widget
#from kivy.properties import NumericProperty, ReferenceListProperty,\
#    ObjectProperty, ListProperty
#from kivy.vector import Vector
from kivy.uix.relativelayout import RelativeLayout
#from kivy.clock import Clock
#import header
from brick_element import Brick
from kivy.factory import Factory


class GameBoard(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super(GameBoard, self).__init__(*args, **kwargs)

        b = Brick()
        self.add_widget(b)


#Builder.load_string(kv_string)
Factory.register('GameBoard', GameBoard)
