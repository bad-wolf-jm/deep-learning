import tensorflow as tf
import sys
from train.optimizers import get_by_id


class CompiledModel(object):
    def __init__(self, file_path):
        script = open(file_path).read()
        self._graph = tf.Graph()
        self._script_globals = globals()
        with self._graph.as_default():
            exec(script, self._script_globals)
            self.name = self._script_globals.get('__model_name__', None)
            self.version = self._script_globals.get('__version__', None)
            self.author = self._script_globals.get('__author__', None)
            self.date = self._script_globals.get('__date__', None)
            self.doc = self._script_globals.get('__doc__', None)
            self.type = self._script_globals.get('__type__', None)
            self.data = self._script_globals.get('__data__', None)
            self.categories = self._script_globals.get('__categories__', None)
            self._input, self._output = self._script_globals['inference']()

    def initialize(self, session=None):
        session.run(tf.global_variables_initializer())

    def save(self, path, session=None):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.save(session, path)

    def restore(self, path, session=None):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver.restore(session, path)

    def initialize_uninitialized_variables(self, session=None):
        global_variables = tf.global_variables()
        is_initialized = session.run([tf.is_variable_initialized(x) for x in global_variables])
        uninitialized_variables = [x for (x, v) in zip(global_variables, is_initialized) if not v]
        if len(uninitialized_variables) > 0:
            session.run(tf.variables_initializer(uninitialized_variables))


class CompiledTrainingModel(CompiledModel):
    def __init__(self, file_path):
        super(CompiledTrainingModel, self).__init__(file_path)
        with self._graph.as_default():
            self._script_globals['loss']()
        self._loss = self._script_globals.get('loss', None)
        self._process_batch = self._script_globals.get('prepare_batch', None)
        self._evaluate_batch = self._script_globals.get('evaluate_batch', None)
        self._optimizer_type = self._script_globals.get('__optimizer__', tf.train.AdamOptimizer)
        self._learning_rate = self._script_globals.get('__learning_rate__', 0.0001)
        self._optimizer_args = self._script_globals.get('__optimizer_args__', {})
        with self._graph.as_default():
            self._optimizer = self._optimizer_type(self._learning_rate, **self._optimizer_args)
            self._train_op = self._optimizer.minimize(self._loss())

    def train(self, train_x, train_y, session=None):
        #print(session)
        session.run(self._train_op, feed_dict = self._process_batch(train_x, train_y))

    def evaluate_batch(self, batch_x, batch_y, session=None):
        d = self._evaluate_batch(batch_x, batch_y, session)
        return d

    def half_learning_rate(self):
        self.train_setup(self._optimizer, self._learning_rate / 2, **self._optimizer_args)


if __name__ == '__main__':
    from models.supervisor_2 import TrainingSupervisor
    x = CompiledTrainingModel('models/bigru_3.py')
    with tf.Session(graph=x._graph) as _session:
        x.train_setup(tf.train.AdamOptimizer, 0.001)
        x.initialize(_session)
        supervisor = TrainingSupervisor(session=_session, model=x, test_interval=1,
                                        validation_interval=1, summary_span=1000)
        for loss in supervisor.run_training():
            print(loss)
    sys.exit(0)