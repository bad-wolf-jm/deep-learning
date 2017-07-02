import sys
import os
import traceback
from multiprocessing import freeze_support
#from pydjay.core.keyboard import key_map

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == '__main__':
    #freeze_support()
    #import pydjay.bootstrap
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.clock import Clock
    from kivy.config import Config
    Config.set('kivy', 'exit_on_escape', '0')
    from ui.main_screen import MainScreen

    Window.clearcolor = (0.1, 0.1, 0.1, 1)
    Window.size = (864, 1252)
    bar = MainScreen()
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
