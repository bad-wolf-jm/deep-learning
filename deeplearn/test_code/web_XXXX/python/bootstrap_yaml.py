import os
import json
import pprint
import glob
import time
import train.graphs as graphs
import train.datasources as datasources
import tensorflow as tf
import yaml


# TODO Class should be able to test if training is resumable
# TODO Class should save the training status and graphs


class CompiledModel(object):
    def __init__(self, id_,  name, model_type, dataset, description="", **hyperparameters):
        super(CompiledModel, self).__init__()
        self.id_=id_
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

    def construct_yaml(self, root):
        self.model_root = os.path.join(root, self.id_)
        self.weight_root = os.path.join(self.model_root, 'weights')
        self.test_root = os.path.join(self.model_root, 'tests')
        for p in [self.model_root, self.weight_root,  # self.train_root, self.notes_root,
                  self.test_root]:
            if not os.path.exists(p):
                os.makedirs(p)
        self.metadata_file = os.path.join(self.model_root, 'model.json')
        self.training_settings_file = os.path.join(self.model_root, 'training.json')

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

    def get_tests(self, min_date=None, max_date=None):
        return self.__get_files(self.test_root, "test-tag*.json", min_date, max_date)

    def get_confusion_matrices(self, min_date=None, max_date=None):
        return self.__get_files(self.test_root, "test-output-matrix*.json", min_date, max_date)

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
        metadata['test_root'] = self.test_root
        metadata['dataset'] = self.dataset_type
        return metadata

    @classmethod
    def compile_yaml(cls, file_name):
        file_content = open(file_name).read()
        model_id = os.path.basename(file_name)
        full_model_description = yaml.load(file_content)
        model_root = os.path.expanduser('~')
        model_root = os.path.join(model_root, '.nn_models', 'bin')
        model_name = full_model_description['name']
        model_type = full_model_description['type']
        model_dataset = full_model_description['dataset']
        model_description = full_model_description['description']
        model_hyperparameters = full_model_description['hyperparameters']
        model_train_settings = full_model_description['training']
        specs = graphs.get_default_model_specs(model_type)
        data_specs = datasources.get_dataset_specs(model_dataset)
        parameters = {name: specs['hyperparameters'][name]['default'] for name in specs['hyperparameters']}
        parameters.update(model_hyperparameters)
        x = cls(model_id, model_name, model_type, model_dataset, model_description, **parameters)
        x.construct_yaml(model_root)
        x.yaml_train_settings = model_train_settings
        x.save_metadata()
        pprint.pprint(parameters)
        return x

    @classmethod
    def load_yaml(cls, file_path):
        file_content = open(file_path).read()
        file_name = os.path.basename(file_path)
        full_model_description = yaml.load(file_content)
        model_root = os.path.dirname(file_name)
        model_root = os.path.join(model_root, '.bin')
        model_root = os.path.join(model_root, file_name)
        if not os.path.exists(model_root):
            return cls.compile_yaml(file_path)
        else:
            m_path = os.path.join(model_root, 'model.json')
            model_metadata_file = open(m_path)
            model_metadata = json.loads(model_metadata_file.read())
            new_instance = cls(model_metadata['name'],
                               model_metadata['type'],
                               model_metadata['dataset'],
                               model_metadata['description'],
                               **model_metadata['hyperparameters'])
            new_instance.metadata_file = m_path
            new_instance.model_root = model_metadata['model_root']
            new_instance.weight_root = model_metadata['weight_root']
            new_instance.test_root = model_metadata['test_root']
            new_instance.yaml_train_settings = full_model_description['training']
            pprint.pprint(model_metadata)
            return new_instance


    def build(self, training=True):
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

    def prefix_exists(self, prefix):
        return len(glob.glob("{p}*".format(p=prefix))) > 0

    def initialize(self):  # , session, training=True, resume=False):
        weight_root = self.weight_root  # if not resume else self.train_root
        with self._graph.as_default():
            if os.path.exists(os.path.join(weight_root, 'model.ckpt.meta')):
                saver = tf.train.Saver()
                saver.restore(self._session, os.path.join(weight_root, "model.ckpt"))
                print('model restored')
            else:
                self._session.run(tf.global_variables_initializer())
                self.save()

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

    def get_weight_file_prefix(self):
        return os.path.join(self.weight_root, "model.ckpt")

    def save_training_state(self, path=None):
        self.save_metadata()
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._session, os.path.join(self.weight_root, "model.ckpt"))

    def restore_last_checkpoint(self):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._session, os.path.join(self.weight_root, "model.ckpt"))

    def load_initial_weights(self, w_prefix):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._session, w_prefix)

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


if __name__ == '__main__':
    fil = os.path.expanduser('~/python/deep-learning/deeplearn/twitter_sentiment/yaml/bigru_cms_user.yaml')
    CompiledModel.load_yaml(fil)  # works
