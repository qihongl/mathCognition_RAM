from matplotlib import pyplot
import matplotlib as mpl
import numpy as np
from getData import getData

showImage = 1

# datasetName = 'oneObj_big'
datasetName = 'multiObj_balanced'
hasLabel = True
img_size1 = 90
img_size2 = 90
batch_size = 10

imgBatch, nextY, coords, _ = getData(batch_size, datasetName, img_size1, img_size2, hasLabel)
for i in xrange(batch_size):
    nextY[i] = int(nextY[i])

print 'image batch dimension:'
print np.shape(imgBatch)

# print 'image indices:'
# print img_idx

# show the 1st image
selectedImgIdx = 1
numObjs = nextY[selectedImgIdx]
print 'number of objects = %d' % (numObjs)

thiscoords = coords[selectedImgIdx]
# thiscoords = np.array(thiscoords)
print thiscoords
print type(thiscoords)
# print np.reshape(thiscoords, [2,numObjs])

if showImage:

    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['black', 'white'])
    tempImg = np.reshape(imgBatch[selectedImgIdx,:], [img_size1, img_size2])
    # tempImg = np.transpose(tempImg)
    fig = pyplot.imshow(tempImg, interpolation='nearest', cmap=cmap)
    pyplot.colorbar(fig, cmap=cmap, ticks=[1, 0])
    pyplot.show()