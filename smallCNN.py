"""
ReLU neurons
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)


Do your imports as usual:
"""

import tensorflow as tf
import numpy as np
from pyprogress import *


# Good old W in a more sophisticated way
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# Good old b in a more sophisticated way
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# W * x
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


# Load your DATA!!
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Start an Interactive session!!!
sess = tf.InteractiveSession()

# PLACEHOLDERS!
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


"""
First convolutional layer
    It convolutes and then does a max pooling
        The convolution computes 32 features for each 5x5 patch
        weight tensor of shape [5,5,1,32]

"""

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# Evolve x into a 4d tensor

x_image = tf.reshape(x, [-1,28,28,1])

# And now convolve x_image with the weight tensor, add the bias,
# apply the ReLU function, and finally max pool it

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# the max_pool_2x2 method reduces the image size to 14x14

"""
Second convolutional layer
    This second layer has 64 features for each 5x5

"""

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# the max_pool_2x2 method reduces the image size to 7x7

"""
Densely Connected layer
    Now that the image size has been reduced to 7x7, we add
    a fully-connected layer with 1024 neurons to allow processing
    the entire image.

    This process, reshapes the tensor from pooling layer
    into a batch of vectors, multiply by a weight matrix,
    add a bias, and apply a ReLU

"""

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


"""
Dropout
    To reduce overfitting, dropout is applied before the readout layer
    Create a placeholder for the probability that a neuron's
    output is kept during dropout
        The advatange of doing this is that the dropout is turned on
        during training and turns off during testing

    No additional scaling needed, tf.nn.dropout does magical stuff

"""

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""
Readout layer
    Back to what we did in digitpredict2

"""

W_fc2  = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

"""
 Now, YOU TRAIN THE MODEL and Evaluate it
  We will replace the steepest gradient descent optimizer with
  the more sophisticated ADAM optimizer.

  We will include the additional parameter keep_prob in
  feed_dict to control the dropout rate.

  We will add logging to every 100th iteration in
  the training process.

"""
# LOSS FUNCTION
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))

# TRAIN steps
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# ARE THE PREDICTIONS TRUE?
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# What is the accuracy then?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all variables
sess.run(tf.global_variables_initializer())

# Load a progress bar for additional fanciness
pb = ProgressBar (20000, name = "Learning Progress", timecount = False, completionprediction = True, colored = False)
pb.begin()
# Let the TRAINING BEGIN
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    pb.inc()
pb.end()

print("test accuracy %g"%accuracy.eval(feed_dict = {
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
