from collections import deque
import numpy as np
import datetime


class StreamSummary(object):
    def __init__(self, summary_span=None, fields=None):
        super(StreamSummary, self).__init__()
        self.current_step = 0
        self._fields = fields
        self._data = {}
        self._created_at = datetime.datetime.today()
        if self._fields is not None:
            for field in self._fields:
                self._data[field] = deque(maxlen=summary_span)

    def add(self, index=0, **kwargs):
        self.current_step = index
        timestamp = (datetime.datetime.today() - self._created_at).total_seconds()
        for field_name in kwargs:
            self._data[field_name].append([timestamp, kwargs[field_name]])

    def get(self, fields=None, min_batch_index=None, max_batch_index=None):
        fields = fields or self._fields
        min_batch_index = min_batch_index or 0
        max_batch_index = max_batch_index or self.current_step

        if min_batch_index is not None and min_batch_index < 0:
            min_batch_index += max_batch_index

        return_value = {}
        for f in fields:
            list_ = self._data[f]
            if len(list_) > 0:
                return_value[f] = [x for x in list_ if x[0] >= min_batch_index and x[0] <= max_batch_index]
            else:
                return_value[f] = []
        return return_value

    def stats(self, fields=None, backlog=None):
        fields = fields or self._fields
        return_value = {}
        for f in fields:
            list_ = self._data[f]
            if len(list_) > 0:
                list_ = [x[1] for x in list_]
                if backlog is not None:
                    list_ = list_[-backlog:]
                return_value[f] = {'mean': np.mean(list_),
                                   'standard_deviation': np.std(list_),
                                   'max': max(list_),
                                   'min': min(list_)}
            else:
                return_value[f] = {'mean': np.nan,
                                   'standard_deviation': np.nan,
                                   'max': np.nan,
                                   'min': np.nan}
        return return_value
