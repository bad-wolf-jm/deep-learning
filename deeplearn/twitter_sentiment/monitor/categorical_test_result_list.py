from kivy.clock import Clock, mainthread
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

#import pydjay.bootstrap
#from behaviors.long_press_button import LongPressButtonBehaviour
from behaviors.track_list_behaviour import TrackListBehaviour
from elements import large_track_list
import widgets
#from pydjay.gui.utils import seconds_to_human_readable
from clickable_area import ImageButton
from elements.simple_detailed_list_item import SimpleDetailedListItem
#from elements.utils import seconds_to_human_readable
#from pydjay.bootstrap import play_queue, playback_manager, session_manager


kv_string = """
#:import label kivy.uix.label
#:import sla kivy.adapters.simplelistadapter
#:import SimpleDetailedListItem elements.simple_detailed_list_item.SimpleDetailedListItem
<MainTrackList>:
    orientation: 'horizontal'
    size_hint: 1,1
    master_list: master_list
    #search_filter:search_filter
    #track_count:track_count
    button_size: 45

    StencilView:
        size_hint: 1,1
        RelativeLayout:
            size: self.parent.size
            pos: self.parent.pos
            BoxLayout:
                orientation: 'vertical'
                size_hint: 1,1
                RelativeLayout:
                    size_hint: 1, None
                    height: 55

                    BoxLayout:
                        orientation: 'horizontal'
                        size_hint: 1, 1
                        padding:[10,0,10,0]
                        spacing: 10
                        canvas.before:
                            Color:
                                rgba: (.3,.3,.3,1) if root.has_focus else (0.1,0.1,0.1,1)
                            Rectangle:
                                pos: self.pos
                                size: self.size
                HDivider:
                LargeTrackList:
                    id: master_list
                    size_hint: 1,1
                    item_class: SimpleDetailedListItem
                    item_convert: root._convert
"""


class MainTrackList(BoxLayout, TrackListBehaviour):
    master_list = ObjectProperty(None)
    preview_player = ObjectProperty(None)
    search_filter = ObjectProperty(None)
    sort = BooleanProperty(True)
    track_list = ObjectProperty(None)
    auto_save = BooleanProperty(False)
    total_track_count = StringProperty("")
    playlist_title = StringProperty("")
    title = StringProperty("")

    def __init__(self, *args, **kwargs):
        super(MainTrackList, self).__init__(*args, **kwargs)
        self._focus = False
        self._keyboard = None
        Clock.schedule_once(self._post_init, -1)
        self._current_selection = None

    def _post_init(self, *args):
        self.adapter = self.master_list.adapter
        self.list_view = self.master_list.list_view

    def _focus_filter(self):
        self.search_filter.focus = True

    def _toggle_keyboard_shortcuts(self, *a):
        if not self.search_filter.focus:
            self.window.request_focus(self)
        else:
            self.window.suspend_focus()
            pass

    def _update_availability(self, *args):
        self.master_list.update_availability(self._track_is_available)

    def _convert(self, row, item):
        return {'row': row,
                'item': item,
                'view': self,
                'drag_context': self.drag_context,
                'is_selected': False}

    def _on_list_touch_down(self, window, event):
        if self.master_list.collide_point(*event.pos):
            if not event.is_mouse_scrolling:
                for data in self.master_list.adapter.data:
                    data['is_selected'] = False

    def set_track_list(self, list, sort=True):
        self.master_list.set_track_list(list, sort, self._track_is_available)
        self._update_track_count()
        num = len(self.master_list.adapter.data)
        time = 0
        if num == 0:
            self.total_track_count = ""
        else:
            for t in self.master_list.adapter.data:
                try:
                    time += t['item'].track.info.length
                except:
                    pass
            self.total_track_count = "[color=#ffffff]" + str(num) + " tracks " + "[/color]" + \
                                     "[color=#444444] | [/color]" + \
                                     "[color=#888888]" + \
                seconds_to_human_readable(time / 1000000000) + "[/color]"

Builder.load_string(kv_string)
Factory.register('MainTrackList', MainTrackList)


if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.clock import Clock
    from kivy.config import Config
    Config.set('kivy', 'exit_on_escape', '0')

    Window.clearcolor = (0,0,0, 1)
    Window.size = (864, 752)
    bar = MainTrackList()
    try:
        runTouchApp(bar)
    except Exception, details:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print '-' * 60
        traceback.print_exc(file=sys.stdout)
        print '-' * 60
        print details

    finally:
        #bar.stop_monitor()
        pass
