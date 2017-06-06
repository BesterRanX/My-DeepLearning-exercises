from BesterTF.Layers import Dense, Slice
import tensorflow as tf


# define data
x_data = [[0, 1], [0, 0], [1, 1], [1, 0]]  # input data

y_data = [[0], [0], [1], [0]]  # target data

test_data = [[1, 1], [0, 1], [1, 0], [1, 1]]

# uninitialised data
input_data = tf.placeholder(tf.float32)
target_data = tf.placeholder(tf.float32)

# define layers
inp = Dense(input_data, input_dim=2, output_dim=6, activation=tf.nn.sigmoid)
h1 = Dense(inp, input_dim=6, output_dim=3, activation=tf.nn.relu)
h2 = Dense(h1, input_dim=3, output_dim=2, activation=tf.nn.relu)
prediction = Dense(h2, input_dim=2, output_dim=1, activation=tf.nn.sigmoid)

# update
loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(target_data, prediction)))
step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialise all variables
init = tf.global_variables_initializer()

# run on the session
with tf.Session() as sess:
    sess.run(init)

    for e in range(500):
        sess.run(step, feed_dict={input_data:x_data, target_data:y_data})

        if  e % 10:
            print("loss", sess.run(loss, feed_dict={input_data:x_data, target_data:y_data}))

    print("\nprediction:")
    print(sess.run(prediction, feed_dict={input_data:x_data, target_data:y_data}))