import sys
import os
import traceback
#from multiprocessing import freeze_support
#from pydjay.core.keyboard import key_map
#sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
print sys.path

if __name__ == '__main__':
    #freeze_support()
    #import pydjay.bootstrap
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.clock import Clock
    from kivy.config import Config
    Config.set('kivy', 'exit_on_escape', '0')
    os.chdir('/home/jalbert/python/deep-learning/deeplearn/twitter_sentiment')
    #os.chdir('/Users/jihemme/python/DJ/deep-learning/deeplearn/twitter_sentiment')
    #from main_screen import MainScreen
    #from stats_graph import TrainingStatsBox
    from stats_display import MainScreen

    Window.clearcolor = (0,0,0, 1)
    Window.size = (864, 752)
    #bar = TrainingStatsBox()
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
        bar.stop_monitor()
        pass
