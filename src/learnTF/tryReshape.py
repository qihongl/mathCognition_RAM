import numpy as np
import tensorflow as tf
#
#
# data = np.reshape(np.arange(18),[2,3,3])
# normLoc = tf.constant(data, name = 'x')
#
# # get the kth glimpose (out of 6)
#
# k  = 1
# imgCoord = tf.Variable(data, name = 'imageCoordinate')
# imgCoord = tf.cast(tf.round(imgCoord), tf.int32)
# imgCoord_r = tf.reshape(imgCoord, [2,9])
#
#
# print tf.shape(imgCoord)
#
# with tf.Session() as session:
#     session.run(tf.initialize_all_variables())
#
#     print(session.run(imgCoord))
#     print(session.run(imgCoord_r))
#
#
#


x = np.zeros(10)

for i in xrange(10):
    x[i] = 1

print x
