# reference: https://github.com/seann999/tensorflow_mnist_ram
import tensorflow as tf
import tf_mnist_loader
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import sys

dataset = tf_mnist_loader.read_data_sets("mnist_data")
save_dir = "/Users/Qihong/Dropbox/github/mathCognition_RAM/src/fixateOneObj/save/"
save_prefix = "save"
#load_path = None
start_step = 0
load_path = save_dir + save_prefix + str(start_step) + ".ckpt"
# to enable visualization, set draw to True
eval_only = 0
animate = 0
draw = 0

# glimpse parameters
minRadius = 4               # zooms -> minRadius * 2**<depth_level>
sensorBandwidth = 8         # fixed resolution of sensor
depth = 3                   # number of zooms
scale = 3
channels = 1                # mnist are grayscale images
totalSensorBandwidth = depth * channels * (sensorBandwidth **2)
nGlimpses = 6               # number of nGlimpses

# number of units
hg_size = 128               # gNet: process glimpse
hl_size = 128               # gNet: process location
g_size = 256                # gNet: produce the "glimpse feature"
cell_size = 256             # coreNet: hidden, input space
cell_out_size = cell_size   # coreNet: hidden, output space

n_classes = 10              # cardinality(Y)
batchSize = 10
learningRate = 1e-3
max_iters = 1000000

mnistSize = 28              # side length of the picture
# pic_ver = 28
# pic_hor = 28

loc_sd = 0.03               # std when setting the location
mean_locs = []              #
sampled_locs = []           # ~N(mean_locs[.], loc_sd)
glimpse_images = []         # to show in window

# set the weights to be small random values, with truncated normal distribution
def weight_variable(paramShape):
    initial = tf.truncated_normal(paramShape, stddev=1.0/paramShape[0]) # for now
    return tf.Variable(initial)

# given the a batch of images and the location, return some glimpses
def glimpseSensor(imgs, normLoc):
    loc = ((normLoc + 1) / 2) * mnistSize        # transform normLoc (-1 to 1) to mnist coordinates
    loc = tf.cast(tf.round(loc), tf.int32)            # round the coordinates
    imgs = tf.reshape(imgs, (batchSize, mnistSize, mnistSize, channels))

    zooms = []          # preallocate for zooms
    # process each image individually
    for k in xrange(batchSize):
        imgZooms = []
        max_radius = minRadius * (scale ** (depth - 1))

        # get one image
        one_img = imgs[k,:,:,:]
        # padding  TODO should we allow the model to move out of the image?
        one_img = tf.image.pad_to_bounding_box(one_img, max_radius, max_radius,
                                               max_radius * 2 + mnistSize, max_radius * 2 + mnistSize)
        #
        one_img = tf.reshape(one_img, (one_img.get_shape()[0].value, one_img.get_shape()[1].value))

        for i in xrange(depth):
            r = int(minRadius * (scale ** (i - 1)))

            d_raw = 2 * r                               # the diameter of the glimpse
            d = tf.constant(d_raw, shape=[1])
            d = tf.tile(d, [2])

            adjusted_loc = max_radius + loc[k,:] - r    # location with the padded mnist reference
            zoom = tf.slice(one_img, adjusted_loc, d)   # crop image to (d x d)

            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
            zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
            imgZooms.append(zoom)

        zooms.append(tf.pack(imgZooms))
        
    zooms = tf.pack(zooms)
    
    glimpse_images.append(zooms)
    
    return zooms


# implements the glimpse network
# the input location norm
def get_glimpse(normLoc):
    # get glimpse using the previous location
    glimpse_input = glimpseSensor(inputs_placeholder, normLoc)
    glimpse_input = tf.reshape(glimpse_input, (batchSize, totalSensorBandwidth))

    # the hidden units that process location & the glimpse
    l_hl = weight_variable((2, hl_size))
    glimpse_hg = weight_variable((totalSensorBandwidth, hg_size))

    hg = tf.nn.relu(tf.matmul(glimpse_input, glimpse_hg))
    hl = tf.nn.relu(tf.matmul(normLoc, l_hl))

    # the hidden units that integrates the location & the glimpses
    hg_g = weight_variable((hg_size, g_size))
    hl_g = weight_variable((hl_size, g_size))

    g = tf.nn.relu(tf.matmul(hg, hg_g) + tf.matmul(hl, hl_g))
    
    return g # glimpse feature


def get_next_input(output, i):
    # the next location is computed by the location network
    mean_loc = tf.tanh(tf.matmul(output, h_l_out))
    mean_locs.append(mean_loc)

    # the actual location used for generating glimpses
    # sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)
    sample_loc = tf.tanh(mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd))
    sampled_locs.append(sample_loc)
    
    return get_glimpse(sample_loc)


def model():
    # initialize the location under unif[-1,1], for all example in the batch
    initial_loc = tf.random_uniform((batchSize, 2), minval=-1, maxval=1)
    # get the glimpse using the glimpse network
    initial_glimpse = get_glimpse(initial_loc)   

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size, state_is_tuple = True)
    initial_state = lstm_cell.zero_state(batchSize, tf.float32)

    inputs = [initial_glimpse]              # save the initial glimpse location
    inputs.extend([0] * (nGlimpses - 1))    # preallocate for the inputs

    #
    outputs, _ = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=get_next_input)
    # get the next location
    get_next_input(outputs[-1], 0)

    return outputs


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  # copied from TensorFlow tutorial
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * n_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


# to use for maximum likelihood with glimpse location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * math.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)


def thresholding(X, upperThreshold, lowerThreshold, shape):
    '''
    Bound the coordinates on the training imgages.
    :param X: the input tensor
    :param upperThreshold: the upper limit allowed
    :param lowerThreshold: the lower limit allowed
    :param shape: the shape of X (read it automatically)
    :return: the input tensor without any "out-of-range" value
    '''
    lower_bound = np.tile(lowerThreshold, shape)    # get_shape
    upper_bound = np.tile(upperThreshold, shape)
    lower_bound = tf.Variable(lower_bound, name='lowerThresholdForMask')
    upper_bound = tf.Variable(upper_bound, name='upperThresholdForMask')
    lower_bound = tf.cast(lower_bound, tf.int32)
    upper_bound = tf.cast(upper_bound, tf.int32)
    X = tf.maximum(X, lower_bound)
    X = tf.minimum(X, upper_bound)
    return X


# TODO implement the hand, then R: handLocation -> {0,1}
def calc_reward(outputs, imgBatch):
    # conside the action at the last time step
    # outputs = outputs[-1] # look at ONLY THE END of the sequence
    # outputs = tf.reshape(outputs, (batchSize, cell_out_size))
    #
    # # the hidden layer for the action network
    # h_a_out = weight_variable((cell_out_size, n_classes))
    #
    # # process its output
    # p_y = tf.nn.softmax(tf.matmul(outputs, h_a_out))
    #
    # max_p_y = tf.arg_max(p_y, 1)
    # # the targets
    # correct_y = tf.cast(labels_placeholder, tf.int64)

    ''''''
    # reshape the input images to x-y coordinate form (10,28,28)
    imgs = tf.reshape(imgBatch, [batchSize, mnistSize, mnistSize])
    # get the coordinate for the glimpse (10,6,2)
    glmpCoords = (sampled_locs + 1) / 2 * mnistSize
    glmpCoords = tf.cast(tf.round(glmpCoords), tf.int32)

    # preallocate for R
    R = tf.constant(np.zeros(batchSize), name='R')
    zeroVector = tf.cast(tf.constant(np.zeros(batchSize)), tf.float32)
    batchIdx = tf.constant(np.arange(batchSize), dtype=tf.int32)
    batchIdx = tf.reshape(batchIdx, [batchSize, 1])


    for i in xrange(nGlimpses):  # loop over glimpses
        # get the location coordinates for all examples in the batch
        ithGlmpCoords = tf.slice(glmpCoords, [0, i, 0], [batchSize, 1, 2])
        ithGlmpCoords = tf.reduce_sum(ithGlmpCoords, 1)  # 10 x 1 x 2 -> # 10 x 2
        ithGlmpCoords = tf.concat(1, [batchIdx, ithGlmpCoords])
        # get the pixel values w.r.t the glimpse location
        pixVal = tf.gather_nd(imgs, ithGlmpCoords)
        # if pixvel(glimpse location) > 0, then R = 1, else R = 0
        R += tf.cast(tf.greater(pixVal, zeroVector), tf.float64)

    R = tf.cast(R, tf.float32)
    reward = tf.reduce_mean(R)  # mean reward over batch and nGlimpses
    ''''''
    # sample the location from the gaussian distribution
    p_loc = gaussian_pdf(mean_locs, sampled_locs)
    p_loc = tf.reshape(p_loc, (batchSize, nGlimpses * 2))

    R = tf.reshape(R, (batchSize, 1))
    R = tf.cast(R, tf.float32)
    # 1 means concatenate along the row direction
    # J = tf.concat(1, [tf.log(p_y + 1e-5) * onehot_labels_placeholder, tf.log(p_loc + 1e-5) * R])
    J = tf.concat(1, [tf.log(p_loc + 1e-5) * R])

    # sum the probability of action and location
    J = tf.reduce_sum(J, 1)
    # average over batch
    J = tf.reduce_mean(J, 0)
    cost = -J

    # Adaptive Moment Estimation
    # estimate the 1st and the 2nd moment of the gradients
    optimizer = tf.train.AdamOptimizer(learningRate)
    train_op = optimizer.minimize(cost)

    # return cost, reward, max_p_y, correct_y, train_op
    return cost, reward, train_op


def evaluate():
    data = dataset.test
    batches_in_epoch = len(data._images) // batchSize
    accuracy = 0

    for i in xrange(batches_in_epoch):
        nextX, nextY = dataset.test.next_batch(batchSize)
        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY,
                     onehot_labels_placeholder: dense_to_one_hot(nextY)}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r

    accuracy /= batches_in_epoch
    print("ACCURACY: " + str(accuracy))


'''
"Main"
'''

with tf.Graph().as_default():
    # the input x, y and yhat
    labels = tf.placeholder("float32", shape=[batchSize, n_classes])
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batchSize, mnistSize * mnistSize), name="images")
    labels_placeholder = tf.placeholder(tf.float32, shape=(batchSize), name="labels")
    onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(batchSize, 10), name="oneHotLabels")

    #
    h_l_out = weight_variable([cell_out_size, 2])
    loc_mean = weight_variable([batchSize, nGlimpses, 2])

    # query the model ouput
    outputs = model()

    # convert list of tensors to one big tensor
    sampled_locs = tf.concat(0, sampled_locs)
    sampled_locs = tf.reshape(sampled_locs, (batchSize, nGlimpses, 2))
    mean_locs = tf.concat(0, mean_locs)
    mean_locs = tf.reshape(mean_locs, (batchSize, nGlimpses, 2))

    glimpse_images = tf.concat(0, glimpse_images)

    #
    # cost, reward, predicted_labels, correct_labels, train_op = calc_reward(outputs, inputs_placeholder)
    cost, reward, train_op = calc_reward(outputs, inputs_placeholder)

    tf.scalar_summary("reward", reward)
    tf.scalar_summary("cost", cost)
    summary_op = tf.merge_all_summaries()

    # initialize the model
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        # if eval_only:
        #     evaluate()
        # else:
        summary_writer = tf.train.SummaryWriter("summary", graph=sess.graph)

        if draw:
            fig = plt.figure()
            txt = fig.suptitle("-", fontsize=36, fontweight='bold')
            plt.ion()
            plt.show()
            plt.subplots_adjust(top=0.7)
            plotImgs = []

        # training
        for step in xrange(start_step + 1, max_iters):
            start_time = time.time()

            # get the next batch of examples
            nextX, nextY = dataset.train.next_batch(batchSize)
            feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY,
                         onehot_labels_placeholder: dense_to_one_hot(nextY)}
            # fetches = [train_op, cost, reward, predicted_labels, correct_labels, glimpse_images]
            fetches = [train_op, cost, reward, glimpse_images]
            # feed them to the model
            results = sess.run(fetches, feed_dict=feed_dict)
            # _, cost_fetched, reward_fetched, prediction_labels_fetched, \
            # correct_labels_fetched, f_glimpse_images_fetched = results

            _, cost_fetched, reward_fetched, f_glimpse_images_fetched = results

            duration = time.time() - start_time

            ''''''
            # sess.run(sampled_locs)
            #
            # sampled_locs_x = sess.run(sampled_locs)
            # sampled_locs_y = sess.run(sampled_locs[0,:,1])
            #
            # # print sampled_locs_x
            sys.exit('STOP HERE')
            ''''''
            if step % 20 == 0:
                ''''''
                if step % 1000 == 0:
                    saver.save(sess, save_dir + save_prefix + str(step) + ".ckpt")
                    if step % 5000 == 0:
                        evaluate()

                ##### DRAW WINDOW ################
                # steps, THEN batch
                f_glimpse_images = np.reshape(f_glimpse_images_fetched, (nGlimpses + 1, batchSize, depth,
                                                                         sensorBandwidth, sensorBandwidth))
                if draw:
                    if animate:
                        fillList = False
                        if len(plotImgs) == 0:
                            fillList = True

                        # display first img in the mini-batch
                        nCols = 4
                        # display the entire image
                        whole = plt.subplot2grid((3, nCols), (0, 1), rowspan=3, colspan=3)
                        whole = plt.imshow(np.reshape(nextX[0,:], [mnistSize,mnistSize]),
                                           cmap=plt.get_cmap('gray'), interpolation="nearest")
                        # TODO
                        # plt.plot(sampled_locs[0,:,0], sampled_locs[0,:,1], '-o', color='lawngreen')
                        whole.autoscale()
                        fig.canvas.draw()


                        # display the glimpses
                        for i in xrange(nGlimpses):
                            # txt.set_text('FINAL PREDICTION: %i\nTRUTH: %i\nSTEP: %i/%i'
                            #              % (prediction_labels_fetched[0], correct_labels_fetched[0], (i + 1), nGlimpses))

                            txt.set_text('Fixation \nSTEP: %i/%i'
                                         % ((i + 1), nGlimpses))

                            for x in xrange(depth):
                                plt.subplot(depth, nCols, 1+nCols*x)
                                if fillList:
                                    plotImg = plt.imshow(f_glimpse_images[i, 0, x], cmap=plt.get_cmap('gray'), interpolation="nearest")
                                    plotImg.autoscale()
                                    plotImgs.append(plotImg)
                                else:
                                    plotImgs[x].set_data(f_glimpse_images[i, 0, x])
                                    plotImgs[x].autoscale()
                            fillList = False

                            fig.canvas.draw()
                            time.sleep(0.1)
                            plt.pause(0.0001)

                    else:
                        # txt.set_text('PREDICTION: %i\nTRUTH: %i' % (prediction_labels_fetched[0], correct_labels_fetched[0]))
                        for x in xrange(depth):
                            for i in xrange(nGlimpses):
                                plt.subplot(depth, nGlimpses, x * nGlimpses + i + 1)
                                plt.imshow(f_glimpse_images[i, 0, x], cmap=plt.get_cmap('gray'),
                                           interpolation="nearest")

                        plt.draw()
                        time.sleep(0.05)
                        plt.pause(0.0001)

                ################################

                print('Step %d: cost = %.5f reward = %.5f (%.3f sec)' % (step, cost_fetched, reward_fetched, duration))

                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

