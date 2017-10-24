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

    def __init__(self, weights_path=None, make_dict=False):
        if weights_path is None:
            path = inspect.getfile(AlexNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'bvlc_alexnet.npy')
            weights_path = path

        self.data_dict = np.load(weights_path, encoding='latin1').item()
        self.imagenet_mean = np.mean([123.68, 116.779, 103.939])  # imagenet mean (channel-wise to global)
        self.names = ['input', 'rgb_scaled', 'bgr_normed',
                      'conv1/lin', 'conv1/relu', 'lrn1', 'pool1',
                      'conv2/lin', 'conv2/relu', 'lrn2', 'pool2',
                      'conv3/lin', 'conv3/relu',
                      'conv4/lin', 'conv4/relu',
                      'conv5/lin', 'conv5/relu', 'pool5',
                      'fc6/lin', 'fc6/relu',
                      'fc7/lin', 'fc7/relu',
                      'fc8/lin', 'fc8/relu',
                      'softmax']
        self.tensors = dict() if make_dict else None
        self.make_dict = make_dict

    def build(self, rgb, rescale=1.0):

        rgb_scaled = tf.multiply(rgb, rescale, name='rgb_scaled')

        rgb_normed = rgb_scaled - self.imagenet_mean
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_normed)
        bgr_normed = tf.concat(axis=3, values=[blue, green, red], name='bgr_normed')

        # conv1
        conv1, _ = self.convolution(bgr_normed, s_h=4, s_w=4, group=1, name='conv1', padding='VALID')
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='lrn1')
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        # conv2
        conv2, _ = self.convolution(maxpool1, s_h=1, s_w=1, group=2, name='conv2')
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='lrn2')
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        # conv3
        conv3, _ = self.convolution(maxpool2, s_h=1, s_w=1, group=1, name='conv3')

        # conv4
        conv4, _ = self.convolution(conv3, s_h=1, s_w=1, group=2, name='conv4')

        # conv5
        conv5, _ = self.convolution(conv4, s_h=1, s_w=1, group=2, name='conv5')
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        # flatten
        # noinspection PyTypeChecker
        maxpool5_flat = tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))], name='pool5_flat')

        # fc6
        relu6, _ = self.fc_layer(in_tensor=maxpool5_flat, name='fc6')
        relu7, _ = self.fc_layer(in_tensor=relu6, name='fc7')
        relu8, _ = self.fc_layer(in_tensor=relu7, name='fc8')

        # prob
        softmax = tf.nn.softmax(relu8, name='softmax')

        if self.make_dict:
            self.tensors['rgb'] = rgb
            self.tensors['rgb_scaled'] = rgb_scaled
            self.tensors['bgr_normed'] = bgr_normed
            self.tensors['pool1'] = maxpool1
            self.tensors['pool2'] = maxpool2
            self.tensors['pool5'] = maxpool5
            self.tensors['softmax'] = softmax

        self.data_dict = None

    def convolution(self, in_tensor, s_h, s_w, group, name, padding='SAME'):
        """From https://github.com/ethereon/caffe-tensorflow
        """
        with tf.variable_scope(name):
            assert isinstance(self.data_dict, dict)
            kernel = tf.constant(self.data_dict[name][0], name='filter')
            biases = tf.constant(self.data_dict[name][1], name='biases')
            if group == 1:
                conv = tf.nn.conv2d(in_tensor, kernel, [1, s_h, s_w, 1], padding=padding)
            else:
                input_groups = tf.split(in_tensor, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
                                 for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)
            conv_lin = tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:], name='lin')
            conv = tf.nn.relu(conv_lin, name='relu')

            if self.make_dict:
                self.tensors[name + '/lin'] = conv_lin
                self.tensors[name + '/relu'] = conv
            return conv, conv_lin

    def fc_layer(self, in_tensor, name):
        with tf.variable_scope(name):
            assert isinstance(self.data_dict, dict)
            weights = tf.constant(self.data_dict[name][0], name='weights')
            biases = tf.constant(self.data_dict[name][1], name='biases')

            fc = tf.nn.bias_add(tf.matmul(in_tensor, weights), biases, name='lin')
            relu = tf.nn.relu(fc, name='relu')

            if self.make_dict:
                self.tensors[name + '/lin'] = fc
                self.tensors[name + '/relu'] = relu
            return relu, fc

    def build_partial(self, in_tensor, input_name, output_name=None, rescale=1.0):

        if 'lin' in input_name:
            in_tensor = tf.nn.relu(in_tensor)
            input_name = input_name.replace('lin', 'relu')

        names_to_build = [n for n in self.names if 'lin' not in n]
        assert input_name in names_to_build

        output_name = output_name or 'softmax'
        if 'lin' in output_name:
            assert output_name.startswith('conv')
            lin_out_option = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            lin_idx = int(output_name[4])
            lin_out_option[lin_idx] = 1
            output_name = output_name.replace('lin', 'relu')
        else:
            lin_out_option = (0, 0, 0, 0, 0, 0, 0, 0, 0)
        assert output_name in names_to_build

        build_ops = list()
        build_ops.append(lambda x: tf.multiply(x, rescale, name='rgb_scaled'))

        def rgb2bgr(x):
            rgb_normed = x - self.imagenet_mean
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_normed)
            return tf.concat(axis=3, values=[blue, green, red], name='bgr_normed')

        build_ops.append(rgb2bgr)

        # conv1
        build_ops.append(lambda x: self.convolution(x, s_h=4, s_w=4, group=1, name='conv1')[lin_out_option[1]])
        build_ops.append(lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05,
                                                                      beta=0.75, bias=1.0, name='lrn1'))
        build_ops.append(lambda x: tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                                  padding='VALID', name='pool1'))

        # conv2
        build_ops.append(lambda x: self.convolution(x, s_h=1, s_w=1, group=2, name='conv2')[lin_out_option[2]])
        build_ops.append(lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05,
                                                                      beta=0.75, bias=1.0, name='lrn2'))
        build_ops.append(lambda x: tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                                  padding='VALID', name='pool2'))

        # conv3
        build_ops.append(lambda x: self.convolution(x, s_h=1, s_w=1, group=1, name='conv3')[lin_out_option[3]])

        # conv4
        build_ops.append(lambda x: self.convolution(x, s_h=1, s_w=1, group=2, name='conv4')[lin_out_option[4]])

        # conv5
        build_ops.append(lambda x: self.convolution(x, s_h=1, s_w=1, group=2, name='conv5')[lin_out_option[5]])
        build_ops.append(lambda x: tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                                  padding='VALID', name='pool5'))

        # flatten
        # noinspection PyTypeChecker
        build_ops.append(lambda x: self.fc_layer(in_tensor=tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))],
                                                                      name='pool5_flat'),
                                                 name='fc6')[lin_out_option[6]])

        # fc6
        build_ops.append(lambda x: self.fc_layer(in_tensor=x, name='fc7')[lin_out_option[7]])
        build_ops.append(lambda x: self.fc_layer(in_tensor=x, name='fc8')[lin_out_option[8]])

        # prob
        build_ops.append(lambda x: tf.nn.softmax(x, name='softmax'))

        start_idx = names_to_build.index(input_name)
        end_idx = names_to_build.index(output_name)
        build_ops = build_ops[start_idx:end_idx]
        print('building partial alexnet:', names_to_build[start_idx + 1:end_idx + 1])
        temp_tensor = in_tensor
        for op in build_ops:
            temp_tensor = op(temp_tensor)
        out_tensor = temp_tensor

        self.data_dict = None
        return out_tensor
