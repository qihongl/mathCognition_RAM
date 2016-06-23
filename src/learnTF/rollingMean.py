import numpy as np
import tensorflow as tf

n = 100
bound = 10

data = np.random.randint(bound, size = n)
trueMean = np.mean(data)
print 'the true mean is %f' % (trueMean)

sum = tf.Variable(0, name = 'sum')

model = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(model)
    # accumulate
    for i in xrange(n):
        sum = sum + data[i]
    # compute the mean
    sum = tf.to_float(sum) / n
    print(sess.run(sum))
