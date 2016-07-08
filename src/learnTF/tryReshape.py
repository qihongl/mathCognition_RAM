import numpy as np
import tensorflow as tf
import sys

batchSize = 10
nGlimpses = 6

data = []

for i in xrange(nGlimpses):
    newdata = tf.reshape(np.arange(batchSize * 2), [batchSize, 2])
    data.append(newdata)

print data

imgCoord = tf.Variable(data, name = 'imageCoordinate')
imgCoord = tf.concat(1, imgCoord)
print imgCoord


# batchsize, glimpses, x&y
imgCoord_r = tf.reshape(imgCoord, [batchSize,nGlimpses,2])
print imgCoord_r

# sys.exit('STOP')

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    # print(session.run(imgCoord))

    print(session.run(imgCoord_r))




