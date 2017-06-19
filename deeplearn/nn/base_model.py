# Geneeric model class


class BaseModel(object):
    def __init__(self):
        super(BaseModel, self).__init__()

    def __call__(self, x, **kwargs):
        """Calling a model runs the feed-forwa rd pass and makes predictions."""
        raise NotImplementedError()

    def validate(self, batch_x, batch_y, **kwargs):
        """Evaluate the model on batch_x and batch_y"""
        raise NotImplementedError()

    def train(self, batch_x, batch_y, **kwargs):
        """Runs a single training step on batch_x and batch_y, returns a dictionary with the metrics returned
        bu the training step"""
        raise NotImplementedError()
