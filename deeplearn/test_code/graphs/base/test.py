import sys
from train.supervisor import TrainingSupervisor
from train.data import vader_sentiment_training_generator, cms_training_generator
import tensorflow as tf
from graphs.base.tf_session import tf_session

def test_graph_type(graph_type):
    model = graph_type()
    model.build_training_model()
    model.initialize()
    model.train_setup(tf.train.AdamOptimizer, learning_rate=0.001)
    tf_session().run(tf.global_variables_initializer())
    foo = TrainingSupervisor(model, 1)
    data_generator = vader_sentiment_training_generator(batch_size=100, epochs=5, validation_size=100, test_size=100)
    try:
        for loss in foo.run_training(data_generator['train'], data_generator['validation']):
            print (loss)
    except KeyboardInterrupt:
        sys.exit(0)
    print('done')


if __name__ == '__main__':
    start_training()
