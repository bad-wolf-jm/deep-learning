import datetime


class Schedule(object):
    def __init__(self):
        super(Schedule, self).__init__()

    def scheduled_at(self, time):
        return None

    def should_run_now(self):
        return self.scheduled_at(datetime.datetime.today)


class MinuteSchedule(Schedule):
    # Syntax: EVERY MINUTE
    def __init__(self):
        super(MinuteSchedule, self).__init__()
        pass

    def __repr__(self):
        return "EVERY MINUTE"

    def scheduled_at(self, time):
        return True


class HourlySchedule(Schedule):
    # Syntax: EVERY HOUR ON MINUTE <minute>
    # Syntax: EVERY HOUR ON MINUTE <list_of_minutes>
    def __init__(self, minute_list):
        super(HourlySchedule, self).__init__()
        self.minute_list = minute_list

    def __repr__(self):
        return "EVERY HOUR ON MINUTE {}".format(self.minute_list)

    def scheduled_at(self, time):
        return time.time().minute in self.minute_list


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


class WeeklySchedule(Schedule):
    # Syntax: EVERY <weekday> AT <time>
    # Syntax: EVERY <list_of_weekdays> AT <time>
    def __init__(self, weekday, time):
        super(WeeklySchedule, self).__init__
        _days = {
            'MONDAY': 0,
            'TUESDAY': 1,
            'WEDNESDAY': 2,
            'THURSDAY': 3,
            'FRIDAY': 4,
            'SATURDAY': 5,
            'SUNDAY': 6
        }
        self.weekday = _days[weekday]
        self.time = time

    def __repr__(self):
        return "EVERY {weekday} AT {time}".format(weekday=self.weekday, time=self.time)

    def scheduled_at(self, time):
        return (time.weekday() == self.weekday) and (self.time == time.time())


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


if __name__ == '__main__':
    x = datetime.datetime.today().date()
    x = datetime.datetime(year=x.year, month=x.month, day=x.day, hour=0, minute=0, second=0)
    delta = datetime.timedelta(minutes=1)
    y = x
    job = DailySchedule(times_list=[
        datetime.time(9, 23),
        datetime.time(12, 3),
        datetime.time(13, 43),
        datetime.time(23, 56),
        datetime.time(10, 45)
    ])
    #job = MinuteSchedule()
    #job = HourlySchedule(minute_list = [5, 10, 15, 20, 35, 36, 45, 52])
    while y.day == x.day:
        if job.scheduled_at(y):
            print(y, job.scheduled_at(y))
        y += delta
