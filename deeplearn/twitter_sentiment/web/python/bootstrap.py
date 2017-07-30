import os
import json
import pprint
import glob
import time
import datetime
import train.graphs as graphs
import train.datasources as datasources
import tensorflow as tf

APP_ROOT_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn")
APP_MODELS_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn/models")
APP_LOG_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn/logs")

for p in [APP_ROOT_FOLDER, APP_MODELS_FOLDER, APP_LOG_FOLDER]:
    if not os.path.exists(p):
        os.makedirs(p)


def list_model_types():
    keys = sorted(graphs.model_specs.keys())
    types = []
    for key in keys:
        type_desc = {
            'name': graphs.model_specs[key]['name'],
            'display_name': graphs.model_specs[key]['display_name'],
            'description': graphs.model_specs[key]['description'],
            'citation': graphs.model_specs[key]['citation'],
            'input_type': graphs.model_specs[key]['input_type'],
            'type': graphs.model_specs[key]['type'],
            'instance_count': count_models(graphs.model_specs[key]['name'])
        }
        types.append(type_desc)
    return types


def list_model_instances():
    m_list = []
    for name in os.listdir(APP_MODELS_FOLDER):
        path = os.path.join(APP_MODELS_FOLDER, name)
        model_desc = os.path.join(path, 'model.json')
        m_list.append(json.loads(open(model_desc).read()))
    return m_list


def count_models(type):
    folder = os.path.join(APP_MODELS_FOLDER, type)
    print (folder)
    if os.path.exists(folder):
        print('Path Exists', len([x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]))
        return len([x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))])
    return None


def get_type_specs(type_name):
    return graphs.model_specs.get(type_name, None)


def list_type_instances(type_name):
    return []


# TODO Class should be able to test if training is resumable
# TODO Class should save the training status and graphs


class PersistentGraph(object):
    def __init__(self, name, model_type, dataset, description="", **hyperparameters):
        super(PersistentGraph, self).__init__()
        self.type = model_type
        self.name = name
        self.hyperparameters = hyperparameters
        self._graph = None
        self.description = description
        self.dataset_type = dataset
        self.dataset = datasources.get_dataset_specs(dataset)
        self.category_labels = self.dataset.get('category_labels')
        self.hyperparameters['num_classes'] = self.dataset.get('number_of_classes')

    @property
    def graph(self):
        return self._graph

    def construct(self):
        self.model_root = os.path.join(APP_MODELS_FOLDER, self.name)
        self.weight_root = os.path.join(self.model_root, 'weights')
        self.train_root = os.path.join(self.model_root, 'training')
        self.notes_root = os.path.join(self.model_root, 'notes')
        self.test_root = os.path.join(self.model_root, 'tests')
        for p in [self.model_root, self.weight_root, self.train_root, self.notes_root, self.test_root]:
            if not os.path.exists(p):
                os.makedirs(p)
        self.metadata_file = os.path.join(self.model_root, 'model.json')
        self.training_settings_file = os.path.join(self.model_root, 'training.json')

    @classmethod
    def new(cls, name, model_type, dataset_type, description="", **hyper):
        specs = graphs.get_default_model_specs(model_type)
        data_specs = datasources.get_dataset_specs(dataset_type)
        parameters = {name: specs['hyperparameters'][name]['default'] for name in specs['hyperparameters']}
        parameters.update(hyper)
        pprint.pprint(parameters)
        x = cls(name, model_type, dataset_type, description, **parameters)
        x.construct()
        return x

    def __get_files(self, root, template='*.json', min_date=None, max_date=None):
        x = []
        files = [[f, os.stat(f).st_ctime] for f in glob.glob("{root}/{template}".format(root=root, template=template))]
        files = sorted(files, key=lambda x: x[1], reverse=True)
        min_date = min_date or 0
        max_date = max_date or time.time()
        return [f[0] for f in files if f[1] >= min_date and f[1] <= max_date]

    def __count_files(self, root):
        x = 0
        for f in glob.glob(os.path.join(root, "*.json")):
            x += 1
        return x

    def get_notes(self, min_date=None, max_date=None):
        return self.__get_files(self.notes_root, min_date, max_date)

    def get_tests(self, min_date=None, max_date=None):
        return self.__get_files(self.test_root, min_date, max_date)

    def get_confusion_matrices(self, min_date=None, max_date=None):
        return self.__get_files(self.test_root, "test-output-matrix*.json", min_date, max_date)

    def count_notes(self):
        return self.__count_files(self.notes_root)

    def count_tests(self):
        return self.__count_files(self.test_root)

    def save_metadata(self):
        with open(self.metadata_file, 'w') as metadata_file:
            metadata_file.write(json.dumps(self.get_metadata(), indent=4))

    def get_metadata(self):
        metadata = {}
        metadata['type'] = self.type
        metadata['name'] = self.name
        metadata['description'] = self.description
        metadata['hyperparameters'] = self.hyperparameters
        metadata['model_root'] = self.model_root
        metadata['weight_root'] = self.weight_root
        metadata['train_root'] = self.train_root
        metadata['notes_root'] = self.notes_root
        metadata['test_root'] = self.test_root
        metadata['dataset'] = self.dataset_type
        return metadata

    @classmethod
    def load(cls, type_, name):
        model_root = os.path.join(APP_MODELS_FOLDER, name)
        m_path = os.path.join(model_root, 'model.json')
        t_path = os.path.join(model_root, 'train.json')
        model_metadata_file = open(m_path)
        model_metadata = json.loads(model_metadata_file.read())
        new_instance = cls(model_metadata['name'], model_metadata['type'],
                           model_metadata['dataset'], model_metadata['description'],
                           **model_metadata['hyperparameters'])
        new_instance.metadata_file = m_path
        new_instance.training_settings_file = t_path
        new_instance.model_root = model_metadata['model_root']
        new_instance.weight_root = model_metadata['weight_root']
        new_instance.train_root = model_metadata['train_root']
        new_instance.notes_root = model_metadata['notes_root']
        new_instance.test_root = model_metadata['test_root']
        pprint.pprint(model_metadata)
        return new_instance

    def load_train_settings(self):
        try:
            return json.loads(open(self.training_settings_file).read())
        except:
            return None

    def save_train_settings(self, *kwargs):
        f = open(self.training_settings_file, 'w')
        return f.write(json.dumps(kwargs))

    def build(self, session, training=True, resume=False):
        new_graph = tf.Graph()
        new_session = tf.Session(graph=new_graph)
        g = graphs.build_skeleton(self.type, **self.hyperparameters)
        with new_graph.as_default():
            if training:
                g.build_training_model()
            else:
                g.build_inference_model()
        self._model = g
        self._graph = new_graph
        self._session = new_session

    def initialize(self, session, training=True, resume=False):
        weight_root = self.weight_root if not resume else self.train_root
        with self._graph.as_default():
            if os.path.exists(os.path.join(weight_root, 'model.ckpt.meta')):
                saver = tf.train.Saver()
                saver.restore(self._session, os.path.join(weight_root, "model.ckpt"))
                restored_from_file = True
            else:
                self._session.run(tf.global_variables_initializer())
                # self.save_training_state()

    def save(self):
        self.save_metadata()
        # self.save_train_settings()
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._session, os.path.join(self.weight_root, "model.ckpt"))

    def dump_weights(self, path):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._session, path)

    def save_training_state(self):
        self.save_metadata()
        # self.save_train_settings()
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._session, os.path.join(self.train_root, "model.ckpt"))

    def restore_last_checkpoint(self):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._session, os.path.join(self.train_root, "model.ckpt"))

    def restore_weights(self):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._session, os.path.join(self.weight_root, "model.ckpt"))

    def initialize_uninitialized_variables(self):
        global_variables = tf.global_variables()
        is_initialized = self._session.run([tf.is_variable_initialized(x) for x in global_variables])
        uninitialized_variables = [x for (x, v) in zip(global_variables, is_initialized) if not v]
        if len(uninitialized_variables) > 0:
            self._session.run(tf.variables_initializer(uninitialized_variables))
