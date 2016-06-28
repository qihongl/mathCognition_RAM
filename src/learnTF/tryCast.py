import numpy as np
import tensorflow as tf

dim = 10
data = np.zeros(dim)
x = tf.constant(data, name = 'x')
logic = tf.equal(np.ones(dim),np.ones(dim))
logic = tf.cast(logic, tf.float64)
print x
print logic

for i in xrange(3):
    x += np.ones(dim, dtype=bool)

for i in xrange(4):
    x += logic


with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    print(session.run(x))
    print(session.run(logic))


# 1. tensorflow constant can be changed!
# 2. tensorflow as treat "truth" as value of 1!
# 3. tensorflow truth value is not 1! I cannot directly add them without CASTING




