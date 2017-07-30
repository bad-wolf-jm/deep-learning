import copy

from graphs.byte_cnn import ByteCNN
from graphs.bidirectional_gru import Tweet2Vec_BiGRU
from graphs.simple_gru import SimpleGRUClassifier
from graphs.simple_gru_convolution import SimpleGRUClassifierConv


model_specs = {
    "ByteCNN": {
        'name': "ByteCNN",
        'display_name': 'Byte level convolutional network',
        'type': 'classifier',
        'constructor': ByteCNN,
        'description': "Byte level very deep convolutional neural network for text sentiment analysis",
        'citation': {'title': 'Very Deep Convolutional Networks for Text Classification',
                     'date': None,
                     'author': None,
                     'link': None},
        'input_type': 'Byte arrays',
        'hyperparameters': {
            'seq_length': {
                'type': int,
                'display': 'Sequence length',
                'description': "The maximal length of input sequences",
                'default': 512
            },
            'input_depth': {
                'type': int,
                'display': 'Encoding dimension',
                'description': "The encoding dimension for inputs",
                'default': 256
            },
            'num_classes': {
                'type': int,
                'display': 'Number of categories',
                'description': "The number of classifying labels this model outputs",
                'default': 3
            },
            'level_features': {
                'type': list,
                'display': 'Convolutional features',
                'description': "Number of output channels at each level",
                'default': [64, 64, 128, 256, 512]
            },
            'sub_levels': {
                'type': list,
                'display': 'Blocks per level',
                'description': "Number of convolutional blocks contained in each level",
                'default': [2, 2, 2, 2]
            },
            'classifier_layers': {
                'type': list,
                'display': 'Classifier features',
                'description': "Dimensions of the fully connected classifier",
                'default': [4096, 2048, 2048]
            }
        }
    },
    "Tweet2Vec_BiGRU": {
        'name': "Tweet2Vec_BiGRU",
        'display_name': "Bidirectional gated recurrent unit classifier",
        'type': 'classifier',
        'constructor': Tweet2Vec_BiGRU,
        'description': "Byte level bidirectional RNN for text sentiment analysis",
        'citation': None,
        'input_type': 'Byte arrays',
        'hyperparameters': {
            'seq_length': {
                'type': int,
                'display': 'Sequence length',
                'description': "The maximal length of input sequences",
                'default': 512
            },
            'embedding_dimension': {
                'type': int,
                'display': 'Encoding dimension',
                'description': "The encoding dimension for inputs",
                'default': 256
            },
            'hidden_states': {
                'type': int,
                'display': 'Internal state dimension',
                'description': "The dimension of the recurrent unit internal state",
                'default': 128
            },
            'num_classes': {
                'type': int,
                'display': 'Number of categories',
                'description': "The number of classifying labels this model outputs",
                'default': 3
            },
            'classifier_layers': {
                'type': list,
                'display': 'Classifier features',
                'description': "Dimensions of the fully connected classifier",
                'default': [4096, 2048, 2048]
            }
        }
    },
    "SimpleGRUClassifier": {
        'name': "SimpleGRUClassifier",
        'display_name': 'Gated recurrent unit classifier',
        'type': 'classifier',
        'constructor': SimpleGRUClassifier,
        'description': "Byte level RNN for text sentiment analysis",
        'citation': None,
        'input_type': 'Byte arrays',
        'hyperparameters': {
            'seq_length': {
                'type': int,
                'display': 'Sequence length',
                'description': "The maximal length of input sequences",
                'default': 512
            },
            'embedding_dimension': {
                'type': int,
                'display': 'Encoding dimension',
                'description': "The encoding dimension for inputs",
                'default': 256
            },
            'hidden_states': {
                'type': int,
                'display': 'Internal state dimension',
                'description': "The dimension of the recurrent unit internal state",
                'default': 128
            },
            'num_classes': {
                'type': int,
                'display': 'Number of categories',
                'description': "The number of classifying labels this model outputs",
                'default': 3
            },
            'rnn_layers': {
                'type': int,
                'display': 'RNN stack depth',
                'description': "The number of stacked recurrent units",
                'default': 3
            },
            'classifier_layers': {
                'type': list,
                'display': 'Classifier features',
                'description': "Dimensions of the fully connected classifier",
                'default': [4096, 2048, 2048]
            }
        }

    },
    "SimpleGRUClassifierConv": {
        'name': "SimpleGRUClassifierConv",
        'display_name': 'Gated recurrent unit classifier with convolutional layers',
        'type': 'classifier',
        'constructor': SimpleGRUClassifierConv,
        'description': "Byte level RNN for text sentiment analysis (convolutional version)",
        'citation': None,
        'input_type': 'Byte arrays',
        'hyperparameters': {
            'seq_length': {
                'type': int,
                'display': 'Sequence length',
                'description': "The maximal length of input sequences",
                'default': 512
            },
            'embedding_dimension': {
                'type': int,
                'display': 'Encoding dimension',
                'description': "The encoding dimension for inputs",
                'default': 256
            },
            'hidden_states': {
                'type': int,
                'display': 'Internal state dimension',
                'description': "The dimension of the recurrent unit internal state",
                'default': 128
            },
            'num_classes': {
                'type': int,
                'display': 'Number of categories',
                'description': "The number of classifying labels this model outputs",
                'default': 3
            },
            'rnn_layers': {
                'type': int,
                'display': 'RNN stack depth',
                'description': "The number of stacked recurrent units",
                'default': 3
            },
            'convolutional_features': {
                'type': list,
                'description': "Dimensions of the fully connected classifier",
                'default': [32, 64, 128, 512]
            },
            'window_sizes': {
                'type': list,
                'description': "Dimensions of the fully connected classifier",
                'default': [3, 3, 3, 3]
            },
            'pooling_sizes': {
                'type': list,
                'description': "Dimensions of the fully connected classifier",
                'default': [3, 3, 3, 3]
            },
            'pooling_strides': {
                'type': list,
                'description': "Dimensions of the fully connected classifier",
                'default': [2, 2, 2, 2]
            },
            'classifier_layers': {
                'type': list,
                'description': "Dimensions of the fully connected classifier",
                'default': [4096, 2048, 2048]
            }
        }
    }
}


def get_default_model_specs(type):
    return copy.deepcopy(model_specs.get(type, None))


def build_skeleton(type, **init_params):
    specs = get_default_model_specs(type)
    if specs is not None:
        model_params = specs['hyperparameters']
        default_params = {name: model_params[name].get('default', None) for name in model_params}
        default_params.update(init_params)
        return specs['constructor'](**default_params)
