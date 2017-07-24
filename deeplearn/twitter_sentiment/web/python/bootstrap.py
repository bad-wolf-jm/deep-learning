import os
import json
import pprint
import web.python.graphs as graphs
import tensorflow as tf

APP_ROOT_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn")
APP_MODELS_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn/models")
APP_LOG_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn/logs")

for p in [APP_ROOT_FOLDER, APP_MODELS_FOLDER, APP_LOG_FOLDER]:
    if not os.path.exists(p):
        os.makedirs(p)

#"""
#skeleton = {'type': 'ByteCNN',
#            'namespace':"XXX"
#            'hyperparameters': {'input_width': seq_length,
#                                'input_depth': input_depth,
#                                'num_categories': num_categories,
#                                'level_features': level_features,
#                                'sub_levels': sub_levels,
#                                'classifier_layers': classifier_layers}}
#"""

def list_model_types():
    keys = sorted(graphs.model_specs.keys())
    types = []
    for key in keys:
        type_desc = {
            'display_name': graphs.model_specs[key]['display_name'],
            'description': graphs.model_specs[key]['description'],
            'citation': graphs.model_specs[key]['citation'],
            'input_type': graphs.model_specs[key]['input_type'],
            'type': graphs.model_specs[key]['type'],
            'instance_count': None
        }
        types.append(type_desc)
    return types

#, #[graphs.model_specs[key] for key in keys]


#class Session(object):
#    def __init__(self):
#        super(Session, self).__init__()
#        self._g = tf.Graph()
#        self._session = None
#
#    def load_graph(self, model):
#        self._model = model
#
#    def start_training_session(self):
#        with self._g.as_default():
#            self._model.build_training_model()
#        self._session = tf.Session(graph=self._g)
#
#    def start_inference_session(self):
#        with self._g.as_default():
#            self._model.build_inference_model()
#        self._session = tf.Session(graph=self._g)
#
#    def close_session(self):
#        self._session.close()
#
#    @property
#    def session(self):
#        return self._session
#
#    def run(self, *args, **kwargs):
#        return self.session.run(*args, **kwargs)


class PersistentGraph(object):
    def __init__(self, name, type_, **hyperparameters):
        super(PersistentGraph, self).__init__()
        self.type = type_
        self.name = name
        self.hyperparameters = hyperparameters
        self._graph = None

    @property
    def graph(self):
        return self._graph

    def construct(self):
        self.model_root = os.path.join(APP_MODELS_FOLDER, self.name)
        self.weight_root = os.path.join(self.model_root, 'weights')
        self.train_root = os.path.join(self.model_root, 'training')
        for p in [self.model_root, self.weight_root, self.train_root]:
            if not os.path.exists(p):
                os.makedirs(p)
        self.metadata_file = os.path.join(self.model_root, 'model.json')

    def save_metadata(self):
        with open(self.metadata_file, 'w') as metadata_file:
            metadata_file.write(json.dumps(self.get_metadata()))

    def get_metadata(self):
        metadata = {}
        metadata['type'] = self.type_
        metadata['name'] = self.name
        metadata['hyperparameters'] = self.hyperparameters
        metadata['model_root'] = self.model_root
        metadata['weight_root'] = self.weight_root
        metadata['train_root'] = self.train_root
        return metadata

    @classmethod
    def new(cls, name, type_, **hyper):
        x = cls(name, type_, **hyper)
        x.construct()
        return x

    @classmethod
    def load(cls, name):
        model_root = os.path.join(APP_MODELS_FOLDER, name)
        m_path = os.path.join(model_root, 'model.json')
        model_metadata_file = open(m_path)
        model_metadata = json.loads(model_metadata_file.read())
        new_instance = cls(model_metadata['name'],
                           model_metadata['type'],
                           **model_metadata['hyperparameters'])
        new_instance.model_root = model_metadata['model_root']
        new_instance.weight_root = model_metadata['weight_root']
        new_instance.train_root = model_metadata['train_root']
        pprint.pprint(model_metadata)
        return new_instance

    def initialize(self, session, training=True):
        restored_from_file = False
        if os.path.exists(os.path.join(self.weight_root, 'model.ckpt')):
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(self.weight_root, "model.ckpt"))
            restored_from_file = True
        g = graphs.build_skeleton(self.type, **self.hyperparameters)
        if training:
            g.build_training_model()
        else:
            g.build_inference_model()
        if not restored_from_file:
            session.run(tf.global_variables_initializer())
        self._graph = g


if __name__ == '__main__':
    pass
    # create_model("My new model",
    #             'ByteCNN',
    #             seq_length=140,
    #             input_depth=256,
    #             num_categories=5,
    #             level_features=[64, 64, 128, 256, 512],
    #             sub_levels=[2, 2, 2, 2],
    #             classifier_layers=[4096, 2048, 2048])
#
#    load_model('My new model')
