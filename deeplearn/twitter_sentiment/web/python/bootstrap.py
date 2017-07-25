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


class PersistentGraph(object):
    def __init__(self, name, model_type, dataset, **hyperparameters):
        super(PersistentGraph, self).__init__()
        self.type = model_type
        self.name = name
        self.hyperparameters = hyperparameters
        self._graph = None
        self.dataset = dataset

    @property
    def graph(self):
        return self._graph

    def construct(self):
        self.model_root = os.path.join(APP_MODELS_FOLDER, self.type, self.name)
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
    def new(cls, name, model_type, dataset_type, **hyper):
        specs = graphs.get_default_model_specs(model_type)
        parameters = {name: specs['hyperparameters'][name]['default'] for name in specs['hyperparameters']}
        parameters.update(hyper)
        pprint.pprint(parameters)
        x = cls(name, model_type, dataset_type, **parameters)
        x.construct()
        return x

    def __get_files(self, root,  min_date=None, max_date=None):
        x = []
        files = [[f, os.stat(f).st_ctime]for f in glob.glob("{root}/*.json".format(root=root))]
        files = sorted(files, key=lambda x: x[1], reverse=True)
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

    def count_notes(self):
        return self.__count_files(self.notes_root)

    def count_tests(self):
        return self.__count_files(self.test_root)

    def save_metadata(self):
        with open(self.metadata_file, 'w') as metadata_file:
            metadata_file.write(json.dumps(self.get_metadata()))

    def get_metadata(self):
        metadata = {}
        metadata['type'] = self.type
        metadata['name'] = self.name
        metadata['hyperparameters'] = self.hyperparameters
        metadata['model_root'] = self.model_root
        metadata['weight_root'] = self.weight_root
        metadata['train_root'] = self.train_root
        metadata['notes_root'] = self.notes_root
        metadata['test_root'] = self.test_root
        metadata['dataset'] = self.dataset
        return metadata

    @classmethod
    def load(cls, type_, name):
        model_root = os.path.join(APP_MODELS_FOLDER, type_, name)
        m_path = os.path.join(model_root, 'model.json')
        t_path = os.path.join(model_root, 'train.json')
        model_metadata_file = open(m_path)
        model_metadata = json.loads(model_metadata_file.read())
        new_instance = cls(model_metadata['name'], model_metadata['type'],
                           model_metadata['dataset'],
                           **model_metadata['hyperparameters'])
        new_instance.metadata_file = m_path
        new_instance.training_settings_file = t_path
        new_instance.model_root = model_metadata['model_root']
        new_instance.weight_root = model_metadata['weight_root']
        new_instance.train_root = model_metadata['train_root']
        new_instance.notes_root = model_metadata['notes_root']
        new_instance.test_root = model_metadata['test_root']
        new_instance.dataset = model_metadata['dataset']
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

    def initialize(self, session, training=True, resume=False):
        new_graph = tf.Graph()
        new_session = tf.Session(graph=new_graph)
        restored_from_file = False
        weight_root = self.weight_root if not resume else self.train_root
        g = graphs.build_skeleton(self.type, **self.hyperparameters)
        with new_graph.as_default():
            if training:
                g.build_training_model()
            else:
                g.build_inference_model()
            if os.path.exists(os.path.join(weight_root, 'model.ckpt.meta')):
                with new_graph.as_default():
                    saver = tf.train.Saver()
                    saver.restore(new_session, os.path.join(weight_root, "model.ckpt"))
                    restored_from_file = True
            else:
                new_session.run(tf.global_variables_initializer())
        self._model = g
        self._graph = new_graph
        self._session = new_session

    def save(self):
        self.save_metadata()
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._session, os.path.join(self.weight_root, "model.ckpt"))

    def save_training_state(self):
        self.save_metadata()
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._session, os.path.join(self.train_root, "model.ckpt"))


if __name__ == '__main__':
    pass
    import numpy as np
    test_data = np.random.randint(1, 256, [1000, 140])
    dummy_output = np.random.rand(1000, 1)

    print('Model Created')
    p = PersistentGraph.new(name="Test Model", type_="ByteCNN")
    p.initialize(session=None)
    foo = p._model.test(test_data, dummy_output, session=p._session)
    model_outputs = [[x[2]] for x in foo['output']]
    p.save()

    print('Model loading')
    q = PersistentGraph.load(name="Test Model", type_="ByteCNN")
    q.initialize(session=None, resume=False)
    foo = q._model.test(test_data, model_outputs, session=q._session)
    print(foo)

    # for i, real, pred in foo['output']:
    #    print(real, pred)
    #model_outputs = [[x[2]] for x in foo['output']]
    # q.save()

#        from train.supervisor import TrainingSupervisor
#        from train.data import sentiment_training_generator, cms_training_generator
#        print('Model reloaded')
#        foo = TrainingSupervisor(q._graph, 10)
#        supervisor = foo
#        data_generator = sentiment_training_generator(batch_size=50, epochs=50, validation_size=50)
#        try:
#            foo.run_training(data_generator['train'], data_generator['validation'], session=q._session)  # batch_generator, validation_iterator)
#        except KeyboardInterrupt:
#            #save_before_exiting()
#            q.save()
#            #foo.shutdown()
#            #sys.exit(0)
#        print('done')
#
