import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread
from imagenet_classnames import class_names
from alexnet import AlexNet




train_x = np.zeros((1, 224, 224, 3)).astype(np.float32)
train_y = np.zeros((1, 1000))
xdim = train_x.shape[1:]

################################################################################
# Read Image, and change to BGR

im1 = (imread('example_images/laska.png')[:224, :224, :3]).astype(np.float32)
im2 = (imread('example_images/poodle.png')[:224, :224, :3]).astype(np.float32)

print(im1.min())
print(im1.max())
x = tf.placeholder(tf.float32, (None,) + xdim)
alex_net = AlexNet()
alex_net.build(x)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

t = time.time()
output = sess.run(alex_net.prob, feed_dict={x: [im1, im2]})

# Output:
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print('Image', input_im_ind)
    for idx in range(5):
        print(class_names[inds[-1 - idx]], output[input_im_ind, inds[-1 - idx]])

print(time.time() - t)