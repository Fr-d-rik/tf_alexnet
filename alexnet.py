################################################################################
#
# Uncluttered things a little bit and stored individual layers in class for easier access
# similar to the vgg_net implementation at https://github.com/machrisaa/tensorflow-vgg
# Based on code by:
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
################################################################################
import inspect
import os
import numpy as np
import tensorflow as tf


class AlexNet:

    def __init__(self, weights_path=None):
        if weights_path is None:
            path = inspect.getfile(AlexNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'bvlc_alexnet.npy')
            weights_path = path
            print(path)

        self.data_dict = np.load(weights_path, encoding='latin1').item()
        self.imagenet_mean = np.mean([103.939, 116.779, 123.68])  # imagenet mean (channel-wise go global)
        print("npy file loaded")

    def build(self, rgb):

        norm_rgb = rgb - self.imagenet_mean
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=norm_rgb)
        norm_bgr = tf.concat(axis=3, values=[blue, green, red])

        # conv1
        self.conv1_in = self.convolution(norm_bgr, c_o=96, s_h=4, s_w=4, padding='SAME', group=1, name='conv1')
        self.conv1 = tf.nn.relu(self.conv1_in)

        # lrn1
        self.lrn1 = tf.nn.local_response_normalization(self.conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool1
        self.maxpool1 = tf.nn.max_pool(self.lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        self.conv2_in = self.convolution(self.maxpool1, c_o=256, s_h=1, s_w=1, padding='SAME', group=2, name='conv2')
        self.conv2 = tf.nn.relu(self.conv2_in)

        # lrn2
        self.lrn2 = tf.nn.local_response_normalization(self.conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool2
        self.maxpool2 = tf.nn.max_pool(self.lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        self.conv3_in = self.convolution(self.maxpool2, c_o=384, s_h=1, s_w=1, padding='SAME', group=1, name='conv3')
        self.conv3 = tf.nn.relu(self.conv3_in)

        # conv4
        self.conv4_in = self.convolution(self.conv3, c_o=384, s_h=1, s_w=1, padding='SAME', group=2, name='conv4')
        self.conv4 = tf.nn.relu(self.conv4_in)

        # conv5
        self.conv5_in = self.convolution(self.conv4, c_o=256, s_h=1, s_w=1, padding='SAME', group=2, name='conv5')
        self.conv5 = tf.nn.relu(self.conv5_in)

        # maxpool5
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # flatten
        # noinspection PyTypeChecker
        maxpool5_flat = tf.reshape(self.maxpool5, [-1, int(np.prod(self.maxpool5.get_shape()[1:]))])

        # fc6
        self.fc6 = self.fc_layer(in_tensor=maxpool5_flat, name='fc6')

        # fc7
        self.fc7 = self.fc_layer(in_tensor=self.fc6, name='fc7')

        # fc8
        self.fc8 = self.fc_layer(in_tensor=self.fc7, name='fc8')

        # prob
        self.prob = tf.nn.softmax(self.fc8)

    def convolution(self, in_tensor, c_o, s_h, s_w, padding, group, name):
        """From https://github.com/ethereon/caffe-tensorflow
        """
        c_i = in_tensor.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        with tf.variable_scope(name):
            assert isinstance(self.data_dict, dict)
            kernel = tf.constant(self.data_dict[name][0])
            biases = tf.constant(self.data_dict[name][1])
            # print('layer: ' + name)
            # print('filter: ' + str(self.data_dict[name][0].shape))
            # print('bias: ' + str(self.data_dict[name][1].shape))
            if group == 1:
                conv = tf.nn.conv2d(in_tensor, kernel, [1, s_h, s_w, 1], padding=padding)
            else:
                input_groups = tf.split(in_tensor, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
                                 for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

    def fc_layer(self, in_tensor, name):
        with tf.variable_scope(name):
            assert isinstance(self.data_dict, dict)
            weights = tf.constant(self.data_dict[name][0], name='weights')
            biases = tf.constant(self.data_dict[name][1], name='biases')
            return tf.nn.relu_layer(in_tensor, weights, biases)
