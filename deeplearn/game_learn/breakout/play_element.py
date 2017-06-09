from kivy.properties import NumericProperty, BooleanProperty, ObjectProperty, ListProperty
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle

class Paddle(Widget):
    color          = ListProperty([1,1,1,.2])
    border_color   = ListProperty([1,1,1,1])
    #remaining_hits = NumericProperty(1)
    #max_hits       = NumericProperty(1, allownone = True)

    def __init__(self, *args, **kwargs):
        super(Paddle, self).__init__(*args, **kwargs)
        self._redraw = Clock.create_trigger(self.draw_window)
        self.bind(size = self._redraw, pos = self._redraw, color = self._redraw)
        self.size_hint = None, None
        self.size = 200, 20
        self.pos  = 200,200
        #self.remaining_hits = self.max_hits

    def draw_window(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*self.color)
            radius = self.height / 2
            RoundedRectangle(size = (self.width, self.height), pos = self.pos, radius = [(radius, radius), (radius, radius), (radius, radius), (radius, radius)])
            Color(*self.border_color)
            Line(rounded_rectangle = [self.x, self.y, self.width, self.height, radius], width = 2)

class Ball(Widget):
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
