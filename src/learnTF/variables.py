import numpy as np
import tensorflow as tf


data = np.random.uniform(-1, 1, size = [4,3,2])
normLoc = tf.constant(data, name = 'x')
print data

data = (data + 1) / 2 * 28
data = np.round(data)
print data

imgCoord = tf.Variable((normLoc+1)/2 * 28, name = 'imageCoordinate')
imgCoord = tf.cast(tf.round(imgCoord), tf.int32)

print tf.shape(imgCoord)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    temp = session.run(imgCoord)
    print(session.run(imgCoord))

R = np.zeros(10)
print R
print type(temp)



# x = tf.Variable(0, name='x')
#
# with tf.Session() as session:
#     for i in range(5):
#         session.run(tf.initialize_all_variables())
#         x = x + 1
#         print x.eval()