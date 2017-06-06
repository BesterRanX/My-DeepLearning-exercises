import tensorflow as tf


def add_layer(inputs, in_size=0, out_size=0, activation=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_uniform([in_size, out_size], -1, 1), name='Weights')

        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')

        Wx_plus_b = tf.matmul(inputs, Weights, name='Wx_plus_b') + biases

        if activation is None:
            return Wx_plus_b
        else:
            return activation(Wx_plus_b)

# define data
x_data = [[0, 1], [0, 0], [1, 1], [1, 0]]  # input data

y_data = [[1.], [0.], [0.], [1.]]  # target data

test_data = [[1, 1], [0, 1], [1, 0], [1, 1]]

# uninitialised data
xs = tf.placeholder(tf.float32, name='x_input')

# normalized = tf.nn.l2_normalize(xs, dim=1)

# add input layer
l1 = add_layer(xs, in_size=2, out_size=10, activation=tf.nn.sigmoid)

# add hidden layer 1
h1 = add_layer(l1, in_size=10, out_size=10, activation=tf.nn.relu)

# add output layer
prediction = add_layer(h1, in_size=10, out_size=1, activation=tf.nn.sigmoid)

with tf.name_scope('update'):
    # calculate the loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_data, prediction)), name='loss')

    # minimize the loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1, name='train').minimize(loss)

# initialise all declared variables of tensorflow
init = tf.global_variables_initializer()

# start to compute operations
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # record trainings
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logs', sess.graph)

    # put all initialised variables into session
    sess.run(init)

    # start to train with 2000 epochs
    for i in range(1000):
        # run "train_step", xs has values equal to x_data.
        summary = sess.run(train_step, feed_dict={xs:x_data})

        train_writer.add_summary(summary, i)

        #print loss each 50 loop
        if i % 100:
            print("loss:", sess.run(loss, feed_dict={xs:x_data}))

    print("\nprediction:")

    # print prediction. predict with "x_data".
    print(sess.run(prediction, feed_dict={xs:test_data}))

# the prediction of test data should be:
# ~0
# ~0
# ~0
# ~0