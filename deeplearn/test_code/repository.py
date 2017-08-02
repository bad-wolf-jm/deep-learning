import os


"""
SHELF_ROOT/<family>/metadata.json            <---- hypermarameters for the whole family
                   /<instance>/metadata.json <---- training parameters for this instance
                              /tests/
                              /summaries/
                              /weights/[weight saved here]
"""
train_settings = {
    'optimizer': {
        'name': 'adam',
        'learning_rate': 0.001,
        'optimizer_parameters': {
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 0.00000001
        }
    },
    'validation_interval': 5,
    'test_interval': 1 * 60,
    'e_mail_interval': 1.5 * 3600,
    'summary_span': None,
    'checkpoint_interval': 45 * 60,
    'batch_size': 100,
    'validation_size': 100,
    'test_size': 1000,
    'epochs': 50
}

class ModelRepository(object):
    def __init__(self, root=None):
        super(ModelRepository, self).__init__()
        self._root = root
        if self._root is not None:
            if not os.path.exists(self._root):
                os.makedirs(self._root)

    def list_elements(self):
        for name in os.listdir(self._root):
            metadata_path = os.path.join(self._root, name, 'metadata.json')
            hyperparameters = json.loads(open(metadata_path).read())
            yield ModelFamily(name, hyperparameters=hyperparameters, parent_shelf=self)


class ModelFamily(object):
    def __init__(self, name=None, hyperparameters=None, repository=None):
        super(ModelFamily, self).__init__()
        self._repository = repository
        self._name = name
        self._hyperparameters = hyperparameters


class ModelInstance(object):
    def __init__(self, name=None, root=None):
        super(ModelFamily, self).__init__()
        self._family = family
        self._name = name
