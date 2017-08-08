import argparse
from web.python.bootstrap import PersistentGraph
from web.python.training import PersistentTrainingSupervisor

from web.python.datasources import generator_specs
from web.python.graphs import model_specs


for m in model_specs:
    for n in generator_specs:





        p = PersistentGraph.load(name="Model_{type}_{name}".format(type=m, name=n), type_=m)
        p.initialize(session=None, training=True, resume=False)
        p.save_metadata()
        train_settings = p.load_train_settings()

        train_settings = {
            'validation_interval': 5,
            'test_interval': 15 * 60,
            'e_mail_interval': 3600,
            'summary_span': None,
            'checkpoint_interval': 30 * 60,
            'batch_size': 100,
            'validation_size': 100,
            'test_size': 1000,
            'epochs': 10
        }

        model_saved_settings = p.load_train_settings()
        model_saved_settings = model_saved_settings or {}
        train_settings.update(model_saved_settings)

        supervisor = PersistentTrainingSupervisor(p,
                                                  validation_interval=train_settings['validation_interval'],
                                                  test_interval=train_settings['test_interval'],
                                                  summary_span=train_settings['summary_span'],
                                                  checkpoint_interval=train_settings['checkpoint_interval'])
        #lock.release()
        print(p.name)
        supervisor.test_train(batch_size=train_settings['batch_size'],
                               validation_size=train_settings['validation_size'],
                               test_size=train_settings['test_size'],
                               epochs=train_settings['epochs'])
