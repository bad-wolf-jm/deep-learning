from parser import ScheduleParser


data = [
    '(EVERY MINUTE)',
    "(EVERY HOUR ON MINUTE (15, 25, 35, 45, 55))",
    "(EVERY MONDAY AT 13:00)",
    "(EVERY TUESDAY AT 15:45)",
    "(EVERY WEDNESDAY AT 15:45)",
    "(EVERY THURSDAY AT 16:45)",
    "(EVERY FRIDAY AT 13:40)",
    "(EVERY SATURDAY AT 01:45)",
    "(EVERY SUNDAY AT 15:04)",
    "(EVERY (MONDAY, WEDNESDAY, FRIDAY) AT 15:04)",
    "(EVERY (MONDAY AT 15:04, WEDNESDAY AT 15:04, FRIDAY AT 15:04))",
    "(EVERY DAY AT 15:45)",
    "(EVERY DAY AT (15:45))",
    "(EVERY DAY AT (15:45, 12:31, 15:34))",
    "(EVERY MONTH ON DAY (1, 3, 5, 7, 9) AT 12:00)",
    #"(STARTING ON 2018-12-12 AT 12:00)",
    """
    # WITH A COMMENT
    (
        STARTING ON 2018-01-01 AT 12:00,
            EVERY 15 MINUTES,
            EVERY 2 DAYS,
            EVERY 3 WEEKS,
            EVERY DAY AT 15:45,
            EVERY DAY AT (15:45),
            EVERY DAY AT (15:45, 12:31, 15:34),
            EVERY MONTH ON DAY (1, 3, 5, 7, 9) AT 12:00,
        ENDING ON 2019-01-01 AT 12:00
    )
    """
]

p = ScheduleParser()

for d in data:
    print('PARSING', d)
    result = p.parse(d)
    print(result, result._starting, result._ending)
