import tensorflow as tf



def Dense(inputs, input_dim=0, output_dim=0, activation=None):
    with tf.name_scope('Layer'):
        Weights = tf.Variable(tf.random_uniform([input_dim, output_dim], -1, 1), name='Weights')

        biases = tf.Variable(tf.zeros([1, output_dim]) + 0.1, name='bias')

        Wx_plus_b = tf.matmul(inputs, Weights, name='Wx_plus_b') + biases

        if activation is None:
            return Wx_plus_b
        else:
            return activation(Wx_plus_b)


def Conv2d(inputs,
           Weights_shape=None,
           bias_value=0.1,
           bias_shape=None,
           strides=None,
           padding='SAME',
           activation=tf.nn.relu):
    # strides [1, x_movement, y_movement, 1]

    Weights = tf.Variable(tf.truncated_normal(Weights_shape, stddev=0.1))

    bias = tf.constant(bias_value, shape=bias_shape)

    convolution = tf.nn.conv2d(inputs, Weights, strides=strides, padding=padding) + bias
    return activation(convolution)


def max_pool_2x2(inputs, ksize=None, strides=None, padding='SAME'):
    return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding)


def Layer(inputs,
          Weights_shape=None,
          bias_value=0.1,
          bias_shape=None,
          activation=None):
    Weights = tf.Variable(tf.truncated_normal(Weights_shape, stddev=0.1))

    bias = tf.constant(bias_value, shape=bias_shape)

    Wx_plus_b = tf.matmul(inputs, Weights) + bias

    if activation is None:
        return tf.Variable(Wx_plus_b)
    else:
        return activation(Wx_plus_b)


def Dropout(inputs, keep_prob=1.0):
    return tf.nn.dropout(inputs, keep_prob=keep_prob)