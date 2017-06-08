from kivy.properties import NumericProperty, BooleanProperty, ObjectProperty, ListProperty
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle

class Brick(Widget):
    color          = ListProperty([1,0,0,.6])
    border_color   = ListProperty([1,0,0,1])
    remaining_hits = NumericProperty(1)
    max_hits       = NumericProperty(1, allownone = True)

    def __init__(self, *args, **kwargs):
        super(Brick, self).__init__(*args, **kwargs)
        self._redraw = Clock.create_trigger(self.draw_window)
        self.bind(size = self._redraw, pos = self._redraw, color = self._redraw, remaining_hits = self._redraw)
        self.size_hint = None, None
        self.size = 100, 30
        self.pos  = 200,200
        self.remaining_hits = self.max_hits

    def draw_window(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*self.color)
            RoundedRectangle(size = (self.width, self.height), pos = self.pos, radius = [(5,5), (5,5), (5,5), (5,5)])
            Color(*self.border_color)
            Line(rounded_rectangle = [self.x, self.y, self.width, self.height, 5], width = 2)


class RedBrick(Brick):
    def __init__(self, *args, **kwargs):
        self.color        = [1,0,0,.6]
        self.border_color = [1,0,0,1]
        self.max_hits     = 4


class GreenBrick(Brick):
    def __init__(self, *args, **kwargs):
        self.color        = [0,1,0,.6]
        self.border_color = [0,1,0,1]
        self.max_hits     = 3


class BlueBrick(Brick):
    def __init__(self, *args, **kwargs):
        self.color        = [0,0,1,.6]
        self.border_color = [0,0,1,1]
        self.max_hits     = 2


class YellowBrick(Brick):
    def __init__(self, *args, **kwargs):
        self.color        = [1,0,0,.6]
        self.border_color = [1,0,0,1]
        self.max_hits     = 1


class WallBrick(Brick):
    def __init__(self, *args, **kwargs):
        self.color        = [1,1,1,.6]
        self.border_color = [.5,.5,.5,1]
        self.max_hits     = None
