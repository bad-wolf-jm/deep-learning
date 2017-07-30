import logging


class LoggingMixin(object):
    def __init__(self, *args, **kwargs):
        super(LoggingMixin, self).__init__()
        self._log_name = type(self).__name__
        self._logger = logging.getLogger(self._log_name)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)


    def log_debug(self, message):
        self._logger.debug(message)

    def log_info(self, message):
        self._logger.info(message)
        pass

    def log_warning(self, message):
        self._logger.warning(message)
        pass

    def log_error(self, message):
        self._logger.error(message)
        pass

    def log_critical(self, message):
        self._logger.critical(message)


if __name__ == '__main__':
    log = LoggingMixin()
    log.log_debug("Debug")
    log.log_info("info")
    log.log_warning('warning')
    log.log_error('error')
    log.log_critical('critical')
