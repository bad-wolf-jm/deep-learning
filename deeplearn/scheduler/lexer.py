import ply.lex as lex


class ScheduleLexer(object):
    tokens = (
        'NUMBER', 'SPACE', 'NEWLINE', 'COMMENT', 'EVERY', 'AT',
        'ON', 'WEEKDAY', 'MINUTE', 'HOUR', 'DAY', 'MONTH',
    )

    literals = [':', '-', '(', ')', ' ', ',']

    t_WEEKDAY = r'SUNDAY|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY'

    t_EVERY = r'EVERY'
    t_AT = r'AT'
    t_ON = r'ON'

    t_MINUTE = r'MINUTE'
    t_HOUR = r'HOUR'
    t_DAY = r'DAY'
    t_MONTH = r'MONTH'

    def __init__(self):
        super(ScheduleLexer, self).__init__()

    def t_NUMBER(self, t):
        r'\d\d+'
        t.value = int(t.value)
        return t

    def t_SPACE(self, t):
        r'\ |\t'
        pass

    def t_NEWLINE(self, t):
        r'\n'
        pass

    def t_COMMENT(self, t):
        r'\#.*\n'
        pass

    def build(self, **kwargs):
        """Create a lexer."""
        self.lexer = lex.lex(
            module=self,
            **kwargs
        )
