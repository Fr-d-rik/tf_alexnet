################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
# Uncluttered things a little bit and stored individual layers in class for easier access
################################################################################

import numpy as np
import time
from scipy.misc import imread
import tensorflow as tf

from caffe_classes import class_names

train_x = np.zeros((1, 227, 227, 3)).astype(np.float32)
train_y = np.zeros((1, 1000))
xdim = train_x.shape[1:]
# ydim = train_y.shape[1]

################################################################################
# Read Image, and change to BGR


im1 = (imread('laska.png')[:, :, :3]).astype(np.float32)
im1 = im1 - np.mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread('poodle.png')[:, :, :3]).astype(np.float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]

net_data = np.load('bvlc_alexnet.npy', encoding='latin1').item()
assert isinstance(net_data, dict)


def conv(input, kernel, biases, c_o, s_h, s_w, padding='VALID', group=1):
    """From https://github.com/ethereon/caffe-tensorflow
    """
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


x = tf.placeholder(tf.float32, (None,) + xdim)

# conv1
conv1W = tf.Variable(net_data['conv1'][0])
conv1b = tf.Variable(net_data['conv1'][1])
conv1_in = conv(x, conv1W, conv1b, c_o=96, s_h=4, s_w=4, padding='SAME', group=1)
conv1 = tf.nn.relu(conv1_in)

# lrn1
lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

# maxpool1
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# conv2
conv2W = tf.Variable(net_data['conv2'][0])
conv2b = tf.Variable(net_data['conv2'][1])
conv2_in = conv(maxpool1, conv2W, conv2b, c_o=256, s_h=1, s_w=1, padding='SAME', group=2)
conv2 = tf.nn.relu(conv2_in)

# lrn2
lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

# maxpool2
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# conv3
conv3W = tf.Variable(net_data['conv3'][0])
conv3b = tf.Variable(net_data['conv3'][1])
conv3_in = conv(maxpool2, conv3W, conv3b, c_o=384, s_h=1, s_w=1, padding='SAME', group=1)
conv3 = tf.nn.relu(conv3_in)

# conv4
conv4W = tf.Variable(net_data['conv4'][0])
conv4b = tf.Variable(net_data['conv4'][1])
conv4_in = conv(conv3, conv4W, conv4b, c_o=384, s_h=1, s_w=1, padding='SAME', group=2)
conv4 = tf.nn.relu(conv4_in)

# conv5
conv5W = tf.Variable(net_data['conv5'][0])
conv5b = tf.Variable(net_data['conv5'][1])
conv5_in = conv(conv4, conv5W, conv5b, c_o=256, s_h=1, s_w=1, padding='SAME', group=2)
conv5 = tf.nn.relu(conv5_in)

# maxpool5
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# fc6
fc6W = tf.Variable(net_data['fc6'][0])
fc6b = tf.Variable(net_data['fc6'][1])

# noinspection PyTypeChecker
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

# fc7
fc7W = tf.Variable(net_data['fc7'][0])
fc7b = tf.Variable(net_data['fc7'][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

# fc8
fc8W = tf.Variable(net_data['fc8'][0])
fc8b = tf.Variable(net_data['fc8'][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# prob
prob = tf.nn.softmax(fc8)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

t = time.time()
output = sess.run(prob, feed_dict={x: [im1, im2]})
################################################################################

# Output:


for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print('Image', input_im_ind)
    for i in range(5):
        print(class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]])

print(time.time() - t)
