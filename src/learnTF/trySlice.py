import numpy as np
import tensorflow as tf


data = np.reshape(np.arange(120),[10,6,2])
normLoc = tf.constant(data, name = 'x')

# get the kth glimpose (out of 6)

k  = 1
imgCoord = tf.Variable(data, name = 'imageCoordinate')
imgCoord = tf.cast(tf.round(imgCoord), tf.int32)

kthCoord = tf.slice(imgCoord, [0,0,0], [10,k,2])

kthCoord_c = tf.reduce_sum(kthCoord, 1)

print tf.shape(imgCoord)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    print(session.run(imgCoord))
    print(session.run(kthCoord))
    print(session.run(kthCoord_c))


