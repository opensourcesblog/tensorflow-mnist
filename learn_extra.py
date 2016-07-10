import tensorflow as tf
import cv2
import numpy as np
from scipy import ndimage
import sys
import os
import math

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def get_x_by_image(folder,image,reverse=False):
    # read the image
    gray = cv2.imread(folder+"/"+image, 0)

    # rescale it
    if reverse:
        gray = cv2.resize(255 - gray, (28, 28))
    else:
        gray = cv2.resize(gray, (28, 28))
    # better black and white version
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """
    flatten = gray.flatten() / 255.0
    return flatten

def get_y_by_digit(digit):
    arr = np.zeros((10))
    arr[digit] = 1
    return arr

def get_learning_batch(folder,reverse=False):
    batch_xs = []
    batch_ys = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            digit = file[-5:-4]
            y = get_y_by_digit(digit)
            x = get_x_by_image(folder,file,reverse=reverse)
            batch_xs.append(x)
            batch_ys.append(y)
    return batch_xs, batch_ys

"""
a placeholder for our image data:
None stands for an unspecified number of images
784 = 28*28 pixel
"""
x = tf.placeholder("float", [None, 784])

# we need our weights for our neural net
W = tf.Variable(tf.zeros([784,10]))
# and the biases
b = tf.Variable(tf.zeros([10]))

"""
softmax provides a probability based output
we need to multiply the image values x and the weights
and add the biases
(the normal procedure, explained in previous articles)
"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""
y_ will be filled with the real values
which we want to train (digits 0-9)
for an undefined number of images
"""
y_ = tf.placeholder("float", [None,10])

"""
we use the cross_entropy function
which we want to minimize to improve our model
"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

"""
use a learning rate of 0.01
to minimize the cross_entropy error
"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


image = sys.argv[1]
train = False if len(sys.argv) == 2 else sys.argv[2]
checkpoint_dir = "cps/"

saver = tf.train.Saver()
sess = tf.Session()
# initialize all variables and run init
sess.run(tf.initialize_all_variables())

folder = sys.argv[1]

# Here's where you're restoring the variables w and b.
# Note that the graph is exactly as it was when the variables were
# saved in a prior training run.
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print 'No checkpoint found'
    exit(1)

if len(sys.argv) > 2:
    reverse =sys.argv[2]
else:
    reverse = False

batch_xs, batch_ys = get_learning_batch(folder,reverse=reverse)
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
saver.save(sess, checkpoint_dir+'model.ckpt')



