import ply.yacc as yacc

import datetime
from schedule_types import (Start, End, ListSchedule, IntervalSchedule, MinuteSchedule,
                            HourlySchedule, DailySchedule, WeeklySchedule, MonthlySchedule)
from lexer import ScheduleLexer


class ScheduleParser(object):
    """docstring for ScheduleParser."""

    def __init__(self):
        super(ScheduleParser, self).__init__()
        self.lexer = ScheduleLexer()
        self.lexer.build()
        self.tokens = self.lexer.tokens
        self.parser = yacc.yacc(
            module=self
        )

    def parse(self, expression):
        return self.parser.parse(expression)

    def p_schedule(self, t):
        """
        schedule : list
        """
        if isinstance(t[1], list) and len(t[1]) == 1:
            t[0] = t[1][0]
        else:
            t[0] = ListSchedule(schedules=t[1])

    def p_schedule_element(self, t):
        """
        schedule_element : schedule_prefix schedule_spec
        """
        schedule_prefix = t[1]
        if schedule_prefix == 'EVERY':
            t[0] = t[2]
        elif schedule_prefix == 'STARTING':
            t[0] = Start(t[2])
        else:
            t[0] = End(t[2])

    def p_schedule_spec(self, t):
        """
        schedule_spec : every_week
                      | every_day
                      | every_hour
                      | every_month
                      | every_minute
                      | every_interval
                      | schedule_date
        """
        t[0] = t[1]

    def p_schedule_prefix(self, t):
        """
        schedule_prefix : EVERY
                        | STARTING
                        | ENDING
        """
        t[0] = t[1]

    def p_schedule_date(self, t):
        """
        schedule_date : ON date AT time
        """
        t[0] = datetime.datetime(
            year=t[2].year,
            month=t[2].month,
            day=t[2].day,
            hour=t[4].hour,
            minute=t[4].minute
        )

    def p_every_interval(self, t):
        """
        every_interval : NUMBER INTERVAL
        """
        t[0] = IntervalSchedule(t[1], t[2])

    def p_every_minute(self, t):
        """
        every_minute : MINUTE
        """
        t[0] = MinuteSchedule()

    def p_every_month(self, t):
        """
        every_month : MONTH ON DAY list AT time
        """
        assert isinstance(t[6], datetime.time)
        days = t[4]
        for x in days:
            assert isinstance(x, int)

        t[0] = MonthlySchedule(
            days=t[4],
            time=t[6]
        )

    def p_every_week(self, t):
        """
        every_week : WEEKDAY AT time
                   | list AT time
                   | list
        """
        if len(t) > 2:
            time = t[3]
            assert isinstance(t[3], datetime.time)
            if isinstance(t[1], str):
                t[0] = WeeklySchedule(
                    weekday=[t[1]],
                    time=time
                )
            else:
                assert isinstance(t[1], list)
                t[0] = WeeklySchedule(
                    weekday=t[1],
                    time=time
                )
        else:
            e = []
            for day, time in t[1]:
                e.append(
                    WeeklySchedule([day], time)
                )
            t[0] = ListSchedule(e)

    def p_every_hour(self, t):
        """
        every_hour : HOUR ON MINUTE list
        """
        t[0] = HourlySchedule(
            minute_list=t[4]
        )

    def p_every_day(self, t):
        """
        every_day : DAY AT list
                  | DAY AT time
        """
        times = t[3]
        if isinstance(t[3], list):
            for x in times:
                assert isinstance(x, datetime.time)
            t[0] = DailySchedule(
                times_list=t[3]
            )
        else:
            assert isinstance(t[3], datetime.time)
            t[0] = DailySchedule(
                times_list=[t[3]]
            )

    def p_list(self, t):
        """
        list : '(' list_of_times ')'
             | '(' list_of_numbers ')'
             | '(' list_of_weekdays ')'
             | '(' list_of_weekdays_w_times ')'
             | '(' list_of_schedules ')'
        """
        t[0] = t[2]

    def p_list_of_schedules(self, t):
        """
        list_of_schedules : list_of_schedules ',' schedule_element
                          | schedule_element
        """
        if len(t) == 2:
            t[0] = [t[1]]
        else:
            t[1].append(t[3])
            t[0] = t[1]

    def p_list_of_weekdays_w_times(self, t):
        """
        list_of_weekdays_w_times : list_of_weekdays_w_times ',' WEEKDAY AT time
                                 | WEEKDAY AT time
        """
        if len(t) == 4:
            t[0] = [(t[1], t[3])]
        else:
            t[1].append((t[3], t[5]))
            t[0] = t[1]

    def p_list_of_weekdays(self, t):
        """
        list_of_weekdays : list_of_weekdays ',' WEEKDAY
                         | WEEKDAY
        """
        if len(t) == 2:
            t[0] = [t[1]]
        else:
            t[1].append(t[3])
            t[0] = t[1]

    def p_list_of_numbers(self, t):
        """
        list_of_numbers : list_of_numbers ',' NUMBER
                        | NUMBER
        """
        if len(t) == 2:
            t[0] = [t[1]]
        else:
            t[1].append(t[3])
            t[0] = t[1]

    def p_list_of_times(self, t):
        """
        list_of_times : list_of_times ',' time
                      | time
        """
        if len(t) == 2:
            t[0] = [t[1]]
        else:
            t[1].append(t[3])
            t[0] = t[1]

    def p_time(self, t):
        """
        time : NUMBER ':' NUMBER
        """
        t[0] = datetime.time(hour=t[1], minute=t[3])

    def p_date(self, t):
        """
        date : NUMBER '-' NUMBER '-' NUMBER
        """
        t[0] = datetime.date(year=t[1], month=t[3], day=t[5])
