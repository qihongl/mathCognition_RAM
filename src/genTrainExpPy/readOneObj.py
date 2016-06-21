import scipy.io
from matplotlib import pyplot
import matplotlib as mpl
import os
import numpy as np

# parameters
datapath = '../../plots/oneObj'
dataset = 'oneObj'
format = '.mat'

showImage = False
batchSize = 5

# count how many training images we have in the directory
files_mat = [i for i in os.listdir(datapath) if i.endswith(format)]
numExamples = len(files_mat)

# take a random batch from the training images
egIdx = np.random.choice(numExamples, batchSize, replace=False)

# read the batch
for i in egIdx:
    fname = dataset + ("%.3d" % (i)) + format
    file = scipy.io.loadmat(datapath + '/' + fname)
    print fname
    # get the values of the image
    img = file['img']


if showImage:
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['white', 'black'])
    fig = pyplot.imshow(img,interpolation='nearest',cmap = cmap)
    pyplot.colorbar(fig,cmap=cmap,ticks=[0,1])
    pyplot.show()