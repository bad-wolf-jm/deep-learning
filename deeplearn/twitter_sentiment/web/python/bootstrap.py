import os
import json
import pprint
import web.python.graphs as graphs


APP_ROOT_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn")
APP_MODELS_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn/models")
APP_LOG_FOLDER = os.path.expanduser("~/.sentiment-analysis/nn/logs")

for p in [APP_ROOT_FOLDER, APP_MODELS_FOLDER, APP_LOG_FOLDER]:
    if not os.path.exists(p):
        os.makedirs(p)

"""
skeleton = {'type': 'ByteCNN',
            'namespace':"XXX"
            'hyperparameters': {'input_width': seq_length,
                                'input_depth': input_depth,
                                'num_categories': num_categories,
                                'level_features': level_features,
                                'sub_levels': sub_levels,
                                'classifier_layers': classifier_layers}}
"""


def create_model(name, type_, **hyperparameters):
    metadata = {}
    metadata['type'] = type_
    metadata['name'] = name
    metadata['hyperparameters'] = hyperparameters
    model_root = os.path.join(APP_MODELS_FOLDER, name)
    weight_root = os.path.join(model_root, 'weights')
    train_root = os.path.join(model_root, 'training')
    for p in [model_root, weight_root, train_root]:
        if not os.path.exists(p):
            os.makedirs(p)
    metadata['model_root'] = model_root
    metadata['weight_root'] = weight_root
    metadata['train_root'] = train_root

    m_path = os.path.join(model_root, 'model.json')
    model_metadata_file = open(m_path, 'w')
    model_metadata_file.write(json.dumps(metadata))


def load_model(name):
    model_root = os.path.join(APP_MODELS_FOLDER, name)
    m_path = os.path.join(model_root, 'model.json')
    model_metadata_file = open(m_path)
    model_metadata = json.loads(model_metadata_file.read())
    g = graphs.build_skeleton(model_metadata['type'], **model_metadata['hyperparameters'])
    print(g)
    pprint.pprint(model_metadata)


if __name__ == '__main__':
    create_model("My new model",
                 'ByteCNN',
                 seq_length=140,
                 input_depth=256,
                 num_categories=5,
                 level_features=[64, 64, 128, 256, 512],
                 sub_levels=[2, 2, 2, 2],
                 classifier_layers=[4096, 2048, 2048])

    load_model('My new model')
