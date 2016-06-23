# Initialize session
import tensorflow as tf
# sess = tf.InteractiveSession()

with tf.Session() as sess:

    # Some tensor we want to print the value of
    x = tf.constant([1.0, 3.0])

    x = tf.reshape(x, [2,1])

    print x
    # Add print operation
    pt = tf.Print(x, [x], message="x = ")

    pt.eval()


    # Add more elements of the graph using a
