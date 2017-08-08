import argparse
from web.python.bootstrap import PersistentGraph

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

p = PersistentGraph.new(name=flags.model_name, model_type=flags.model_type, dataset_type=flags.dataset_type)
p.initialize(session=None)
p.save_metadata()
