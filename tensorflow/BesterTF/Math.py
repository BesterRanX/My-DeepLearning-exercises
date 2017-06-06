import tensorflow as tf


def accuracy(prediction=None, target_data=None):
    correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(target_data, 1))

    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))