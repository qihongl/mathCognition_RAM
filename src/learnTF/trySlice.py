import numpy as np
import tensorflow as tf

# batchSize = 10
# imgSize = 3
# n_glimpses = 7
#
# data = np.reshape(np.arange(n_glimpses * batchSize),[n_glimpses,batchSize,1])
# imgs = tf.Variable(data, name = 'images')
# imgs = tf.cast(imgs, tf.int32)
#
#
# slice = tf.slice(imgs, [n_glimpses-1,0,0], [1,10,1])
#
# with tf.Session() as session:
#     session.run(tf.initialize_all_variables())
#     print(session.run(imgs))
#     print(session.run(slice))



numObjs = 5

data = np.arange(numObjs)
var = tf.Variable(data, name = 'images')
var = tf.cast(var, tf.int32)


slice = tf.slice(var, [0], [3])

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    print(session.run(var))
    print(session.run(slice))