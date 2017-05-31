import tensorflow as tf


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)


with tf.Session() as ses:
    print(ses.run(output, feed_dict={input1:[5], input2:[6]}))