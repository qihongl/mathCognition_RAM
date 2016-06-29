import numpy as np
import tensorflow as tf
import sys

picSize = 3
batchSize = 5
lower_bound = 0
upper_bound = 40

lower_bound = np.tile(lower_bound, [batchSize, picSize,picSize])
upper_bound = np.tile(upper_bound, [batchSize, picSize,picSize])

lower_bound = tf.Variable(lower_bound, name = 'lowerThresholdForMask')
upper_bound = tf.Variable(upper_bound, name = 'upperThresholdForMask')

img = np.reshape(np.arange(batchSize * picSize**2), [batchSize,picSize,picSize])
img = img - 1
img = tf.Variable(img, name = 'imageBatch')

# smallValMask = tf.greater(lower_bound,img)
# largeValMask = tf.greater(img, upper_bound)
img = tf.maximum(img, lower_bound)
img = tf.minimum(img, upper_bound)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    print sess.run(img)


# img = tf.cast(img,tf.int32)

#
# smallVals = tf.boolean_mask(img, smallValMask, name='boolean_mask')
# largeVals = tf.boolean_mask(img, largeValMask, name='boolean_mask')
#
# temp = tf.sparse_mask(img, smallValMask, name='coordinates_smaller_than_lb')
#
#
#
# # # img = tf.scatter_add(img, smallValMask, smallValMask, use_locking=None, name=None)
# # temp = tf.boolean_mask(img, smallValMask, name='boolean_mask')
# #
# # temp = tf.where(smallValMask)
# # # print img[temp]
# # temp = tf.reshape(temp, tf.shape(temp))
# # sp =  tf.shape(temp)
#
# # sys.exit("STOP HERE")
#
#
# # img = tf.scatter_update(img, temp, 0, use_locking=None, name=None)
#
# img += tf.cast(smallValMask, tf.int32)
# img -= tf.cast(largeValMask, tf.int32)
#
#





