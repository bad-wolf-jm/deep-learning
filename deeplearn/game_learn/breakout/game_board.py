from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from brick_element import Brick, RedBrick, GreenBrick, BlueBrick, YellowBrick
from play_element import Paddle
from kivy.factory import Factory
from kivy.clock import Clock


bricks = {'r':RedBrick,
          'g':GreenBrick,
          'b':BlueBrick,
          'y':YellowBrick}


level = ['rrrrrrrrrr',
         'rrrrrrrrrr',
         'gggggggggg',
         'gggggggggg',
         'bbbbbbbbbb',
         'bbbbbbbbbb',
         'yyyyyyyyyy',
         'yyyyyyyyyy']



class GameBoard(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super(GameBoard, self).__init__(*args, **kwargs)
        self._redraw = Clock.create_trigger(self.draw_window)
        self.bind(size = self._redraw, pos = self._redraw)


        aimg = Image(source='breakout_bg.png', allow_stretch = True, keep_ratio = False)
        self.bricks = []
        self.add_widget(aimg)
        #self.add_widget(b)
        #b = Brick()
        #self.add_widget(b)
        Clock.schedule_once(self._post_init, 0)


    def _post_init(self, *a):
        self.load_level(level)

    def draw_window(self, *a):
        y = self.height - 50
        for row in self.bricks:
            length = len(row)
            x = (self.width - (100 * len(row) + (len(row)-1)*8)) / 2
            for brick in row:
                x += 108
                brick.pos = x,y
            y -= 38


    def load_level(self, level):
        self.bricks = [[bricks[color]() for color in row] for row in level]
        for row in self.bricks:
            for b in row:
                self.add_widget(b)
        self.draw_window()
#        print self.height
#        y = self.height - 100
#        for row in level:
#            length = len(row)
#            x = self.width - (100 * len(row) + (len(row)-1)*5)
#            for brick in row:
#                br = bricks[brick]()
#                self.bricks.append(br)
#                x += 125
#                #self.add_widget(br)
#                br.pos = x,y
#            y -= 35


Factory.register('GameBoard', GameBoard)
