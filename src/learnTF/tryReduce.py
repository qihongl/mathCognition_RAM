import numpy as np
import tensorflow as tf


data = np.reshape(np.arange(27),[3,3,3])
imgCoord = tf.Variable(data, name = 'imageCoordinate')

imgCoord = tf.slice(data, [0,0,0], [1,1,1])
print tf.shape(imgCoord)

imgCoord_reduced = tf.reduce_sum(imgCoord)
print tf.shape(imgCoord)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    print(session.run(imgCoord))
    print(session.run(imgCoord_reduced))




