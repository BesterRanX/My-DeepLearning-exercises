
import tensorflow as tf


counter = tf.Variable(0, name='counter')
one = tf.constant(1)

new_counter = tf.add(counter, 1)

update = tf.assign(counter, new_counter) # assign to counter a 'new counter'

init = tf.global_variables_initializer() # must bei initialized

# session run
with tf.Session() as ses:
    ses.run(init)

    # update 4 times
    for _ in range(4):
        print("counter:", ses.run(update))