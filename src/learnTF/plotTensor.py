'''
Tryh the MNIST tutorial:  https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# load the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



batchSize = 10
picSize = 28

r = 8
seq = np.arange(0,2,.1) * np.pi
numPoints = len(seq)
x = np.sin(seq) * r + picSize/2
y = np.cos(seq) * r + picSize/2


xx = tf.Variable(x, name = 'xCoordinates')
yy = tf.Variable(y, name = 'yCoordinates')

print x

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    batch_xs, _ = mnist.train.next_batch(batchSize)

    img = batch_xs[0,:]
    img = np.reshape(img,[picSize,picSize])
    print session.run(xx)
    xxx = session.run(xx)
    yyy = session.run(yy)


fig, ax = plt.subplots()
ax.imshow(img,cmap=plt.get_cmap('gray'), interpolation="nearest")

ax.plot(xxx,yyy, '-o', color = 'lawngreen')
plt.show()

print('done!')
