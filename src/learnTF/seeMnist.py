'''
Tryh the MNIST tutorial:  https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

# load the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set constants
n = 28 **2
nClasses = 10
maxIter = 1000
learningRate = .1
batchSize = 100;

# we don't need to specifiy the number of exampels...
x = tf.placeholder(tf.float32, [None, n])
# allocate the parameters
W = tf.Variable(tf.zeros([n, nClasses]))
b = tf.Variable(tf.zeros([nClasses]))

# build the model
# yhat = softmax(XW+b)
yhat = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.placeholder(tf.float32, [None, nClasses])

# training
# definte the cross entropy loss
crossEnt = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices = [1]))
# define the training algorithm
train_step = tf.train.AdagradOptimizer(learningRate).minimize(crossEnt)

# initialize the parameters (W and b)
init = tf.initialize_all_variables()
# strat training
session = tf.Session()
session.run(init)

# SGD
for i in range(maxIter):
    batch_xs, batch_yhats = mnist.train.next_batch(batchSize)
    # session.run(train_step, feed_dict= {x:batch_xs, y:batch_yhats} )

print type(batch_xs)
print batch_xs.shape
print batch_xs[0,:].shape
img = batch_xs[0,:]
img = np.reshape(img,[28,28])


# print np.mean(img)
# print '%d (%f) of the pixel values are zeros.' % (np.sum(img == 0), np.sum(img == 0) / 28 **2.0)

fig, ax = plt.subplots()
ax.imshow(img,cmap=plt.get_cmap('gray'), interpolation="nearest")


r = 8
seq = np.arange(0,2,.1) * math.pi
numPoints = len(seq)


x = np.sin(seq) * r + 14
y = np.cos(seq) * r + 14

ax.plot(x,y, '-o', color = 'lawngreen')
# ax.plot([3,20],[3,20], '-o', color = 'lawngreen')
plt.show()



# compute the performance
# hits = tf.equal(tf.argmax(y,1), tf.argmax(yhat,1))
# hits_cast = tf.cast(hits, tf.float32)
# accuracyhat = tf.reduce_mean(hits_cast)

# show final test performance
# accuracy = session.run(accuracyhat, feed_dict={x: mnist.test.images, y: mnist.test.labels})
# print('Final test set accuracy is %f' % (accuracy))

print('done!')
