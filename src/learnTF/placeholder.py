import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
path = '/Users/Qihong/Dropbox/github/mathCognition_RAM/src/learnTF/'
filename = path + "MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
# image_gray = tf.placeholder("uint8", [None, None, 3])
image_gray = tf.reduce_mean(2,image)
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})
    result = session.run(image_gray, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()