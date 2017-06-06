import tensorflow as tf


matrix1 = tf.constant([[4,4],[3,3]])
matrix2 = tf.constant([[4],[2]])

product = tf.matmul(matrix1, matrix2) # equals to numpy.dot(matrix1, matrix2)

with tf.Session() as ses:
    result = ses.run(product)
    print(result)