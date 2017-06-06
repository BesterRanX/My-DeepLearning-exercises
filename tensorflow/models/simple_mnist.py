import tensorflow as tf
from BesterTF.Layers import Dense, Conv2d, max_pool_2x2, Layer, Dropout
from BesterTF.Math import accuracy
from tensorflow.examples.tutorials.mnist import input_data

# data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
dt_input = tf.placeholder(tf.float32, [None, 784]) # image size is 28x28
target_data = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(dt_input, [-1, 28, 28,1])

# layers
# conv1 layer
# 5x5 size, 1 output, 32 height
conv1 = Conv2d(x_image, Weights_shape=[5, 5, 1, 32], bias_shape=[32], strides=[1, 1, 1, 1], activation=tf.nn.relu)
maxpool1 = max_pool_2x2(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

# conv2 layer
# 5x5 size, 1 output, 64 height
conv2 = Conv2d(maxpool1, Weights_shape=[5, 5, 32, 64], bias_shape=[64], strides=[1, 1, 1, 1])
maxpool2 = max_pool_2x2(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

# reshaping to 7x7 size, 64 heights
maxpool2_flat = tf.reshape(maxpool2, [-1, 7*7*64])

# 1024 heights
fc_1 = Layer(maxpool2_flat, Weights_shape=[7*7*64, 1024], bias_shape=[1024], activation=tf.nn.relu)
drop = Dropout(fc_1, 1.0)

# dense layers
prediction = Layer(drop, Weights_shape=[1024, 10], bias_shape=[10], activation=tf.nn.softmax)

# update
cross_entropy = tf.reduce_mean(-tf.reduce_sum(target_data * tf.log(prediction), reduction_indices=[1]))
step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
accuracy = accuracy(prediction, target_data)
    `
# initialise all variables
init = tf.global_variables_initializer()

# run on the session
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

    for e in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        step.run(feed_dict={dt_input:batch_xs, target_data:batch_ys})

        if e % 50:
            print("accuracy", sess.run(accuracy, feed_dict={dt_input:mnist.test.images, target_data:mnist.test.labels}))

    batch_xs, batch_ys = mnist.train.next_batch(200)
    print("\nprediction:")
    print(sess.run(prediction, feed_dict={dt_input:batch_xs, target_data:batch_ys}))