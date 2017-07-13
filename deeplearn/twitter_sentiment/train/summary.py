from collections import deque
import numpy as np


class StreamSummary(object):
    def __init__(self, summary_span=None, fields=None):
        super(StreamSummary, self).__init__()
        self.current_step = 0
        self._fields = fields
        self._data = {}
        if self._fields is not None:
            for field in self._fields:
                self._data[field] = deque(maxlen=summary_span)

    def add(self, index=0, **kwargs):
        self.current_step = index
        for field_name in kwargs:
            self._data[field_name].append([index, kwargs[field_name]])

    def get(self, fields=None, min_batch_index=None, max_batch_index=None):
        fields = fields or self._fields
        min_batch_index = min_batch_index or 0
        max_batch_index = max_batch_index or self.current_step  # max([x[0] for x in self.losses])
        return_value = {}
        for f in fields:
            list_ = self._data[f]  # getattr(self, f)
            if len(list_) > 0:
                return_value[f] = [x for x in list_ if x[0] >= min_batch_index and x[0] <= max_batch_index]
            else:
                return_value[f] = []
        return return_value

    def stats(self, fields=None, backlog=None):
        fields = fields or self._fields
        #backlog = backlog or len(self._data[self._data.keys()])
        return_value = {}
        for f in fields:
            list_ = self._data[f]  # getattr(self, f)
            if len(list_) > 0:
                #print (list_)
                list_ = [x[1] for x in list_]
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


class TrainingSummary(object):
    def __init__(self, summary_span=None, fields=None):
        super(TrainingSummary, self).__init__()
        self.train_summary = StreamSummary(summary_span, fields)
        self.validation_summary = StreamSummary(summary_span, fields)

    def add_to_summary(self, summary, index=0, **kwargs):
        summary_name = summary + '_summary'
        summary = getattr(self, summary_name)
        #print (kwargs)
        summary.add(index=index, **kwargs)

    def get_summary(self, summary, fields=None, min_batch_index=None, max_batch_index=None):
        summary_name = summary + '_summary'
        summary = getattr(self, summary_name)
        return summary.get(fields=fields, min_batch_index=min_batch_index, max_batch_index=max_batch_index)

    def get_stats(self, summary, fields=None, backlog=None):
        summary_name = summary + '_summary'
        summary = getattr(self, summary_name)
        return summary.stats(fields=fields, backlog=backlog)
