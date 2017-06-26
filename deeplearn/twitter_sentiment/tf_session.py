import tensorflow as tf

__session__ = None

def tf_session():
    global __session__
    if __session__ is not None:
        return __session__
    else:
        __session__ = tf.Session()
        return __session__

def run_graph(op_dict, feed_dict):
    keys = [k for k in op_dict]
    operations = [op_dict[key] for key in keys]
    sess_ret_val = tf_session().run(operations, feed_dict = feed_dict)
    return {key:val for key, val in zip(keys, sess_ret_val)}
