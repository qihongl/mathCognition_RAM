import numpy as np
import tensorflow as tf

batchSize = 5
imgSize = 3

data = np.reshape(np.arange(batchSize * (imgSize**2)),[batchSize,imgSize,imgSize])

coords = [[0,0],[0,0],[0,0],[1,1],[1,2]]
print coords


batchIdx = tf.constant(np.arange(batchSize), dtype= tf.int32)
batchIdx = tf.reshape(batchIdx, [5,1])

imgs = tf.Variable(data, name = 'images')
imgs = tf.cast(imgs, tf.int32)
coords = tf.Variable(coords, name = 'pixelCoordinates')
print coords
print batchIdx

coords = tf.concat(1,[batchIdx,coords])
coords = tf.cast(coords, tf.int32)

print coords

targets = tf.gather_nd(imgs, coords)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    print(session.run(imgs))
    # print(session.run(coords))
    print(session.run(targets))





