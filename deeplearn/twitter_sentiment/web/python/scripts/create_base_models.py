import argparse
from web.python.bootstrap import PersistentGraph
from web.python.datasources import generator_specs
from web.python.graphs import model_specs


flags = argparse.ArgumentParser()
flags.add_argument('-n', '--name',
                   dest='model_name',
                   type=str,
                   default='',
                   help='A name for the model')
flags.add_argument('-t', '--type',
                   dest='model_type',
                   type=str,
                   default='',
                   help='The type for the model')
flags.add_argument('-d', '--dataset',
                   dest='dataset_type',
                   type=str,
                   default='',
                   help='The type for the dataset')
flags = flags.parse_args()


for m in model_specs:
    for n in generator_specs:
        p = PersistentGraph.new(name="Model_{type}_{name}".format(type=m, name=n), model_type=m, dataset_type=n)
        p.initialize(session=None)
        p.save_metadata()
