from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty
from kivy.factory import Factory
from kivy.uix.modalview import ModalView

kv_string = """
<SSHConnect>:
    size_hint: .7,None
    height: 450
    #short_list: short_list
    title: "CONNECT TO REMOTE HOST VIA SSH"
    canvas:
        Color:
            rgba: 0.7,0.7,0.7,.98
        Rectangle:
            size: self.size
            pos: self.pos

    BoxLayout:
        orientation: 'vertical'
        #spacing:45
        BoxLayout:
            orientation: 'horizontal'
            size_hint: 1, None
            height: 45
            padding: [10,0,10,0]
            canvas.before:
                Color:
                    rgba: 0.3,0.3,0.3,.98
                Rectangle:
                    size: self.size
                    pos: self.pos

            Label:
                size_hint: 1,1
                height: 25
                font_size: 20
                markup: True
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                text: root.title
        ModalHDivider:
        BoxLayout:
            layout: 'horizontal'
            size_hint: 1,.5
            padding: [25,5,45,5]
            spacing:25
            Image:
                size_hint: None,.75
                width:self.height
                keep_ratio: True
                pos_hint: {'center_y':.5}
                source:'ui/icon-warning.png'
            Label:
                size_hint: 1,1
                height: 25
                font_size: 15
                markup: True
                color: .2,.2,.2,1
                halign: 'justify'
                valign: 'middle'
                text_size: self.size
                text: "Provide the login information for the remote host. If the scripts have to run in a virtual environment, provide its full path ('~' is supported). The virtualenv field will be ignored if left empty. 'root' refers to the root folder where the script's main function resides"
                multiline: True
        BoxLayout:
            size_hint: .75, 1
            orientation: 'vertical'
            padding:[10,10,10,10]
            spacing: 10
            pos_hint: {'center_x':.5}
            LabelledTextInput:
                orientation: 'horizontal'
                size_hint: 1,  None
                height: 30
                spacing: 3
                label_color: .4,.4,.4,1
                bold: True
                label_width:75
                label: "Host:"

            LabelledTextInput:
                orientation: 'horizontal'
                size_hint: 1,  None
                height: 30
                spacing: 3
                label_color: .4,.4,.4,1
                bold: True
                label_width:75
                label: "Username:"

            LabelledTextInput:
                orientation: 'horizontal'
                size_hint: 1,  None
                height: 30
                spacing: 3
                label_color: .4,.4,.4,1
                bold: True
                label_width:75
                label: "Password:"
                password:True

            LabelledTextInput:
                orientation: 'horizontal'
                size_hint: 1, None
                height: 30
                label_color: .4,.4,.4,1
                bold: True
                label_width:75
                label: 'Virtualenv:'

            LabelledTextInput:
                orientation: 'horizontal'
                size_hint: 1, None
                height: 30
                label_color: .4,.4,.4,1
                bold: True
                label_width:75
                label: 'Root:'

        Widget:
            size_hint: None,None
            height: 15
        BoxLayout:
            size_hint: 1, None
            orientation: 'vertical'
            height: 80
            Label:
                size_hint: 1,1
                height: 25
                font_size: 15
                markup: True
                color: .2,.2,.2,1
                halign: 'center'
                valign: 'middle'
                text_size: self.size
                text: 'Tap anywhere outside this dialog to dismiss'

            Button:
                size_hint: None, None
                width: 300
                height: 50
                pos_hint: {'center_x':.5}
                text: 'Connect'
        Widget:
            size_hint: None,None
            height: 10
"""

class SSHConnect(ModalView):
    short_list = ObjectProperty(None)
    item_count = StringProperty("")

    def __init__(self, *args, **kw):
        """Doc."""
        super(SSHConnect, self).__init__(*args, **kw)
        #self._drag_payload = None
        #self.register_event_type('on_playlist_selected')
        #Clock.schedule_once(self._post_init, -1)

    def _post_init(self, *a):
        pass

    def on_playlist_selected(self, *a):
        """Doc."""
        pass

    #def open(self, title, pl_list):
    #    """Doc."""
    #    super(PlaylistSelector, self).open()
    #    self.title = title
    #    self.short_list.set_track_list(pl_list, sort=False)
    #    N = len(pl_list)
    #    if N == 1:
    #        self.item_count = "1 item"
    #    else:
    #        self.item_count = "%s items"%N
    #    self.short_list.set_keyboard_handlers({'enter': self._select_playlist})
    #    self.short_list.focus()

    #def dismiss(self):
    #    """Doc."""
    #    super(PlaylistSelector, self).dismiss()

    #def _select_playlist(self):
    #    """Doc."""
    #    # print self.short_list.current_selection
    #    self.dispatch('on_playlist_selected', self.short_list.current_selection['item'].track)
    #    self.dismiss()

    # def do_filter(self, window, text):
    #    self.short_list.short_list.do_filter(text)

    #def _keyboard_closed(self):
    #    """Doc."""
    #    self._keyboard.unbind(on_key_down = self._on_keyboard_down)
    #    self._keyboard = None

    #def request_focus(self, *a):
    #    """Doc."""
    #    pass

    # def remove_unavailable_tracks(self, *a):
    #    foo = RemoveUnavailableDialog(self)
    #    foo.open()
    #
    #    def do_remove_unavailable_tracks(self):
    #        tracks = [x for x in pydjay.bootstrap.get_short_list() if pydjay.bootstrap.track_is_available(x)]
    #        pydjay.bootstrap.set_short_list(tracks)
    #        self.short_list.focus()


Builder.load_string(kv_string)
#Factory.register('PlaylistSelector', PlaylistSelector)
