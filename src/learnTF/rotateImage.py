import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# First, load the image
path = '/Users/Qihong/Dropbox/github/mathCognition_RAM/src/learnTF/'
filename = path + "MarshOrchid.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')
s = tf.Variable(0, name='s')
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    x = tf.transpose(x, perm=[1, 0, 2])
    x = tf.reverse_sequence(x, [height] * width, 1, batch_dim=0)
    x = tf.transpose(x, perm=[1, 0, 2])
    result = session.run(x)


print(result.shape)
plt.imshow(result)
plt.show()