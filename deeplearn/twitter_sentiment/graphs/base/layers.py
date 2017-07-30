import tensorflow as tf


def var(input_shape, trainable=True, name="variable", scope=None):
    initializer = tf.random_normal(input_shape, stddev=0.35)
    v = tf.Variable(initializer, name=name, trainable=trainable)
    return v


def temporal_batch_normalize(i_tensor):
    mean, variance = tf.nn.moments(i_tensor, axes=[0, 1])
    o_tensor = tf.nn.batch_normalization(
        i_tensor, mean, variance, offset=None, scale=None, variance_epsilon=1e-08)
    return o_tensor


def convolutional_block(i_tensor, i_features, o_features, scope=None):
    with tf.variable_scope(scope):
        kernel = var(input_shape=[3, i_features, o_features], name='layer_1_convolution')
        x = tf.nn.conv1d(i_tensor, kernel, stride=1, padding="SAME")
        x = temporal_batch_normalize(x)
        conv = tf.nn.relu(x)
        kernel = var(input_shape=[3, o_features, o_features], name='layer_2_convolution')
        x = tf.nn.conv1d(conv, kernel, stride=1, padding="SAME")
        x = temporal_batch_normalize(x)
        conv_block = tf.nn.relu(x)
        return conv_block


def max_pool_layer(i_tensor, window_size, strides):
    conv_block_1 = tf.reshape(
        i_tensor, [-1, 1, i_tensor.shape[1].value, i_tensor.shape[2].value])
    conv_block_1 = tf.nn.max_pool(conv_block_1, [1, 1, window_size, 1], [1, 1, strides, 1], 'SAME')
    conv_block_1 = tf.reshape(
        conv_block_1, [-1, conv_block_1.shape[2].value, conv_block_1.shape[3].value])
    return conv_block_1
