import tensorflow as tf
import numpy as np

x = tf.placeholder('float')
y = tf.placeholder('float')

w = tf.Variable([1.0,2.0], name = 'w')
y_model = tf.mul(x, w[0]) + w[1]

error = tf.square(y - y_model)


train_op = tf.train.GradientDescentOptimizer(.01).minimize(error)

init = tf.initialize_all_variables()

errors = []
with tf.Session() as session:
    session.run(init)
    for i in range(500):
        x_train = tf.random_normal((1,), mean=5, stddev=2.0)
        y_train = x_train * 2 + 6
        x_value, y_value = session.run([x_train, y_train])
        _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
        errors.append(error_value)
    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))

import matplotlib.pyplot as plt
plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()
plt.savefig("errors.png")