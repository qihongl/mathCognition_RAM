import numpy as np
import tensorflow as tf


data = np.random.randint(1000, size=10)

x = tf.constant(data, name = 'x')
y = tf.Variable(5*(x**2) - 3*x + 15, name = 'y')

model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))



x = tf.Variable(0, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    for i in range(5):
        session.run(model)
        x = x + 1
        print(session.run(x))