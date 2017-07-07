from collection import deque
import numpy as np


class StreamSummary(object):
    def __init__(self, summary_span=None):
        super(TrainingSummary, self).__init__()
        self.current_step = 0
        self.losses = deque(maxlen=summary_span)
        self.accuracies = deque(maxlen=summary_span)
        self.times = deque(maxlen=summary_span)

    def add(self, index=0, loss=0, accuracy=0, time=0):
        self.current_step = index
        self.losses.append([index, loss])
        self.accuracies.append([index, accuracy])
        self.time.append([index, time])

    def get(self, fields=None, min_batch_index=None, max_batch_index=None):
        fields = fields or ['losses', 'accuracies', 'times']
        min_batch_index = min_batch_index or 0
        max_batch_index = max_batch_index or max([x[0] for x in self.losses])
        return_value = {}
        for f in fields:
            list_ = getattr(self, f)
            if len(list_) > 0:
                return_value[f] = [x for x in list_ if x[0] >= min_batch_index and x[0] <= max_batch_index]
            else:
                return_value[f] = []
        return return_value

    def stats(self, fields=None, backlog=None):
        fields = fields or ['losses', 'accuracies', 'times']
        backlog = backlog or len(self.losses)
        return_value = {}
        for f in fields:
            list_ = getattr(self, f)
            if len(list_) > 0:
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
    def __init__(self,  summary_span=None):
        super(TrainingSummary, self).__init__()
        self.train_sumary = StreamSummary(summary_span)
        self.validation_sumary = StreamSummary(summary_span)

    def add_to_summary(self, summary, index=0, loss=0, accuracy=0, time=0):
        summary_name = summary + '_summary'
        summary = getattr(self, summary_name)
        summary.add(index=index, loss=loss, accuracy=accuracy, time=time)

    def get_summary(self, summary, fields=None, min_batch_index=None, max_batch_index=None):
        summary_name = summary + '_summary'
        summary = getattr(self, summary_name)
        return summary.get(fields=fields, min_batch_index=min_batch_index, max_batch_index=max_batch_index)

    def get_stats(self, summary, fields=None, backlog=None):
        summary_name = summary + '_summary'
        summary = getattr(self, summary_name)
        return summary.stats(fields=fields, backlog=backlog)
