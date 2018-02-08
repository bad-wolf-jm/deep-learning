from parser import ScheduleParser


class Scheduler(Base):

    id = Column(Integer, primary_key=True)

    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)

    # Who created the scheduled job and when
    added_by = Column(Text, nullable=False)
    created_time = Column(DateTime, nullable=False)

    # when the scheduling actually starts
    start_time = Column(DateTime, nullable=False)

    # after this time, the job won't run
    end_time = Column(DateTime, nullable=True)

    # A description of the schedule on which to run this process.  The language is described
    # in the parser module
    schedule_script = Column(Text, nullable=False)

    # PAUSED OR SCHEDULED. Paused jobs don't run until they are unpaused
    status = Column(String(16), nullable=False)

    # path to the script to run
    script = Column(Text, nullable=False)
    script_arguments = Column(Text, nullable=False)

    # this is NULL when the job isn't running
    job_pid = Column(Integer, nullable=True)

    # number of times the script was skipped because it was blocked, or successfully
    # started (independent of whether the run itself was successful).
    skip_count = Column(Integer, nullable=True)
    run_count = Column(Integer, nullable=True)

    _schedule_parser = ScheduleParser()

    @property
    def skip_percentage(self):
        return float(self.skip_count) / (self.skip_count + self.run_count)

    @property
    def schedule(self):
        try:
            return self._schedule_parser.parse(self.schedule_script)
        except:
            return none

    @property
    def is_running(self):
        return (self.job_pid is not None)

    @property
    def is_blocked(self):
        for j in self.blocking_jobs:
            if j.is_running:
                return True
        return False

    @property
    def is_paused(self):
        return self.schedule != 'SCHEDULED'

    def _should_run_now(self):
        return self.schedule_object.scheduled_at(datetime.datetime.today())

    def _can_run_now(self):
        t = datetime.datetime.today()
        if t >= self.start_date:
            if self.end_date is None or t <= self.end_date:
                return (not self.is_running) and \
                            (not self.is_blocked) and \
                                (not self.is_paused)
            else:
                return False
        else:
            return False

    def run(self):
        if self._should_run_now() and self._can_run_now():
            self.job_pid = os.getpid()
            self.run_count += 1
            session.commit()
            # run the job
        else:
            if self._should_run_now():
                self.skip_count += 1
                self.job_pid = None
                session.commit()
