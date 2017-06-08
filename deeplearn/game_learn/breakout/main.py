from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
#from kivy.properties import NumericProperty, ReferenceListProperty,\
#    ObjectProperty, ListProperty
#from kivy.vector import Vector
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import header
import game_board


kv_string = """
<MainScreen>:
    orientation: 'vertical'
    GameHeader:
        size_hint: 1, None
        height:75
    GameBoard:
        size_hint: 1,1
"""

class MainScreen(BoxLayout):
    def __init__(self, *args, **kwargs):
        super(MainScreen, self).__init__(*args, **kwargs)

Builder.load_string(kv_string)


class BreakoutClone(App):
    def build(self):
        game = MainScreen()
        return game

if __name__ == '__main__':
    from kivy.core.window import Window
    from kivy.config import Config
    Config.set('kivy', 'exit_on_escape', '0')
    Window.clearcolor = (0,0,0, 1)
    Window.size = (1200,720)
    BreakoutClone().run()
