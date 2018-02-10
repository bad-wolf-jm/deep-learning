import datetime


class Schedule(object):
    def __init__(self):
        super(Schedule, self).__init__()
        self._starting = None
        self._ending = None

    def scheduled_at(self, time):
        if self._starting is not None and self._ending is not None:
            return time >= self._starting and time <= self._ending

        if self._starting is not None:
            return time >= self._starting

        if self._ending is not None:
            return time <= self._ending

        return True

    def should_run_now(self):
        return self.scheduled_at(datetime.datetime.today)

    def next_n_events(self, n=2, date_start=None):
        date_start = date_start if date_start is not None else datetime.datetime.today()
        next_events = []
        for i, ev in enumerate(self.list_events(start_date)):
            if i < n:
                next_events.append(ev)
            else:
                break
        return next_events

    def list_events(self, date_start):
        if self.scheduled_at(date_start):
            yield date_start
        e = date_start
        while True:
            e = self.first_event_after(e)
            yield e

    def get_events_in_range(self, date_start, date_end):
        for ev in self.list_events(date_start):
            if ev <= date_end:
                yield ev
            else:
                break

    def _round_date(self, d):
        return datetime.datetime(
            year=d.year,
            month=d.month,
            day=d.day,
            hour=d.hour,
            minute=d.minute
        )

    def first_event(self):
        if self.scheduled_at(self._starting):
            return self._starting
        return self.first_event_after(self._starting)

    def first_event_after(self, date):
        raise NotImplemented()


class Start(object):
    # Syntax: STARTING ON date AT time
    def __init__(self, date):
        super(Start, self).__init__()
        self.date = datetime.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=date.hour,
            minute=date.minute
        )


class End(object):
    # Syntax: ENDING ON date AT time
    def __init__(self, date):
        super(End, self).__init__()
        self.date = datetime.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=date.hour,
            minute=date.minute)


class ListSchedule(Schedule):
    # Syntax: (<schedule>, <schedule>, <schedule>,...)
    def __init__(self, schedules):
        super(ListSchedule, self).__init__()
        self.schedules = []  # schedules
        for s in schedules:
            if isinstance(s, Start):
                if self._starting is None:
                    self._starting = s.date
                else:
                    raise Exception()
            elif isinstance(s, End):
                if self._ending is None:
                    self._ending = s.date
                else:
                    raise Exception()
            else:
                self.schedules.append(s)
        for s in self.schedules:
            s._starting = self._starting
            s._ending = self._ending

    def scheduled_at(self, time):
        for s in self.schedules:
            if s.scheduled_at(time):
                return True

    def first_event_after(self, date):
        return min([x.first_event_after(date) for x in self.schedules])

    def __repr__(self):
        return repr(self.schedules)


class IntervalSchedule(Schedule):
    # Syntax: EVERY number unit
    def __init__(self, number, unit):
        super(IntervalSchedule, self).__init__()
        self._number = number
        self._unit = unit

    def __repr__(self):
        return "EVERY {} {}".format(self._number, self._unit)

    @property
    def unit_factor(self):
        return {
            "MINUTES": 60.0,
            "HOURS": 3600.0,
            "DAYS": 24 * 3600.0,
            "WEEKS": 7 * 24 * 3600.0
        }[self._unit]

    def scheduled_at(self, time):
        delta = time - self._starting
        interval = self._number * self.unit_factor
        return (delta.total_seconds() % interval) == 0

    def first_event_after(self, date):
        delta = date - self._starting
        interval = self._number * self.unit_factor
        q = (delta.total_seconds() // interval) + interval
        return self._starting + datetime.timedelta(seconds=q)


class MinuteSchedule(Schedule):
    # Syntax: EVERY MINUTE
    def __init__(self):
        super(MinuteSchedule, self).__init__()
        pass

    def __repr__(self):
        return "EVERY MINUTE"

    def first_event_after(self, date):
        f = self._round_date(date)
        return f + datetime.timedelta(minutes=1)


class HourlySchedule(Schedule):
    # Syntax: EVERY HOUR ON MINUTE <minute>
    # Syntax: EVERY HOUR ON MINUTE <list_of_minutes>
    def __init__(self, minute_list):
        super(HourlySchedule, self).__init__()
        self.minute_list = sorted(minute_list)

    def __repr__(self):
        return "EVERY HOUR ON MINUTE {}".format(self.minute_list)

    def scheduled_at(self, time):
        return time.time().minute in self.minute_list

    def _set_minute(self, date, minute):
        return datetime.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=date.hour,
            minute=minute
        )

    def first_event_after(self, date):
        d = self._round_date(date)
        m = date.time().minute
        for x in self.minute_list:
            if x > m:
                return self._set_minute(d, x)
        else:
            d += datetime.timedelta(hours=1)
            return self._set_minute(d, self.minute_list[0])


class DailySchedule(Schedule):
    # Syntax: EVERY DAY AT <time>
    # Syntax: EVERY DAY AT <list_of_times>
    def __init__(self, times_list):
        super(DailySchedule, self).__init__()
        self.times_list = times_list

    def __repr__(self):
        return "EVERY DAY AT {}".format(self.times_list)

    def scheduled_at(self, time):
        rounded_time = datetime.time(time.time().hour, time.time().minute)
        return rounded_time in self.times_list

    def _set_time(self, date, time):
        return datetime.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=time.hour,
            minute=time.minute
        )

    def first_event_after(self, date):
        d = self._round_date(date)
        m = date.time()
        for x in self.times_list:
            if x > d:
                return self._set_time(d, x)
        else:
            d += datetime.timedelta(days=1)
            return self._set_time(d, self.times_list[0])


class WeeklySchedule(Schedule):
    # Syntax: EVERY <weekday> AT <time>
    # Syntax: EVERY <list_of_weekdays> AT <time>
    # Syntax: EVERY <list_of_(<weekday AT TIME>)>
    def __init__(self, weekday, time):
        super(WeeklySchedule, self).__init__()
        _days = {
            'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3,
            'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
        }
        self.weekday = [_days[w] for w in weekday]
        self.time = time

    def __repr__(self):
        return "EVERY {weekday} AT {time}".format(weekday=self.weekday, time=self.time)

    def scheduled_at(self, time):
        return (time.weekday() in self.weekday) and (self.time == time.time())

    def _set_time(self, date, time):
        return datetime.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=time.hour,
            minute=time.minute
        )

    def first_event_after(self, date):
        #weekdays = [0,1,2,3,4,5]
        day = date.weekday()
        time = date.time()
        if day in self.weekday and time < self.time:
            return self._set_time(date, self.time)
        else:
            for d in self.weekday:
                if d > day:
                    diff_day = d - day
                    delta = datetime.timedelta(days=diff_day)
                    next_date = date + delta
                    return self._set_time(next_date, self.time)
            diff_day = day - self.weekday[0]
            delta = datetime.timedelta(days=diff_day)
            next_date = date + delta
            return self._set_time(next_date, self.time)


class MonthlySchedule(Schedule):
    # Syntax: EVERY MONTH ON DAY <list of days> AT <time>
    def __init__(self, days, time):
        super(MonthlySchedule, self).__init__()
        self.days = days
        self.time = time

    def __repr__(self):
        return "EVERY MONTH ON DAY {days} AT {time}".format(days=self.days, time=self.time)

    def scheduled_at(self, time):
        return (time.day in self.days) and (time.time() == self.time)

    def first_event_after(self, date):
        day = date.day()
        time = date.time()
        if day in self.days and time < self.time:
            return self._set_time(date, self.time)
        else:
            for d in self.days:
                if d > day:
                    diff_day = d - day
                    delta = datetime.timedelta(days=diff_day)
                    next_date = date + delta
                    return self._set_time(next_date, self.time)
            diff_day = day - self.days[0]
            delta = datetime.timedelta(days=diff_day)
            next_date = date + delta
            return self._set_time(next_date, self.time)
