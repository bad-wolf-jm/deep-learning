import tensorflow as tf

optimizer_specs = {
    'gradient_descent': {
        'display_name': 'Stocastic gradient descent',
        'description': 'Optimizer that implements the gradient descent algorithm.',
        'docs': 'https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer',
        'constructor': tf.train.GradientDescentOptimizer,
        'learning_rate': 0.00001,
        'optimizer_parameters': {}
    },
    'adadelta': {
        'display_name': 'ADADELTA method',
        'docs': 'https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer',
        'description':  'The method dynamically adapts over time using only first order information and '
                        'has minimal computational overhead beyond vanilla stochastic gradient descent. '
                        'The method requires no manual tuning of a learning rate and appears robust to '
                        'noisy gradient information, different model architecture choices, various data '
                        'modalities and selection of hyperparameters.',
        'constructor': tf.train.AdadeltaOptimizer,
        'learning_rate': 0.001,
        'optimizer_parameters': {
            'rho': 0.95,
            'epsilon': 1e-08,
        }
    },
    'adagrad': {
        'display_name': 'ADAGRAD optimizer',
        'docs': 'https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer',
        'description': 'Optimizer that implements the Adagrad algorithm.',
        'constructor': tf.train.AdagradOptimizer,
        'learning_rate': 0.00001,
        'optimizer_parameters': {
            'initial_accumulator_value': 0.1
        }
    },
    'adagrad_da': {
        'display_name': 'ADAGRAD dual averaging optimizer',
        'docs': 'https://www.tensorflow.org/api_docs/python/tf/train/AdagradDAOptimizer',
        'description':  'Adagrad Dual Averaging algorithm for sparse linear models. AdagradDA is typically used '
                        'when there is a need for large sparsity in the trained model. This optimizer only '
                        'guarantees sparsity for linear models. Be careful when using AdagradDA for deep '
                        'networks as it will require careful initialization of the gradient accumulators for '
                        'it to train.',
        'constructor': tf.train.AdagradDAOptimizer,
        'learning_rate': 0.00001,
        'optimizer_parameters': {
               'initial_gradient_squared_accumulator_value': 0.1,
               'l1_regularization_strength': 0.0,
               'l2_regularization_strength': 0.0,
        }
    },
    'momentum': {
        'display_name': 'Momentum optimizer',
        'docs': 'https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer',
        'description': 'Optimizer that implements the Momentum algorithm.',
        'constructor': tf.train.MomentumOptimizer,
        'learning_rate': 0.00001,
        'optimizer_parameters': {
            'momentum': 0.1,
            'use_nesterov': False
        }
    },
    'adam': {
        'display_name': 'Adam optimizer',
        'description': 'Optimizer that implements the Adam algorithm.',
        'docs': 'https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer',
        'constructor': tf.train.AdamOptimizer,
        'learning_rate': 0.001,
        'optimizer_parameters': {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 0.00000001
        }
    },
}

def get_by_id(id_):
    return optimizer_specs.get(id_, None)

def build(name, learning_rate=None, **kwargs):
    x = get_by_id(name)
    if x is not None:
        constructor = x['constructor']
        learning_rate = x['learning_rate']
        kw = {}
        kw.update(x['optimizer_parameters'])
        kw.update(kwargs)
        return {'constructor': constructor, 'learning_rate': learning_rate, 'optimizer_args': kw}
    else:
        return None
