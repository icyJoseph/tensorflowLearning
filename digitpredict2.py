"""
Create a softmax regression function that is a model for recognizing
MNIST digits, based on looking at every pixel in the image

Use Tensorflow to train the model to recognize digits by having it
"look" at thousands of examples (and run our first Tensorflow session to do so)

Check the model's accuracy with our test data

Build, train, and test a multilayer convolutional neural network
to improve the results

"""

import tensorflow as tf
import numpy as np
from pyprogress import * # You probably have to pip this one

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Start an Interactive session
sess = tf.InteractiveSession()

"""
  Build a softmax regression model

"""

# Placeholders
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

 # variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable (tf.zeros([10]))

"""
 Why 10?
 There are digits from 0 to 9 as outputs, therefore our vectors
 will point to 10 possible values (classes)

"""
#Initialize all variables
sess.run(tf.global_variables_initializer())

# Predict Class and Loss function
y = tf.matmul(x,W) + b # Regression Model. It only takes one line!

# Loss function, in this case cross-entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

"""
You built the model, now you TRAIN IT!
"""
#This line is one powerful command
#which adds new operations to the computation graph.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#it includes gradient computation, parameter update steps and apply
#updates to the parameters

# Progress Bar
pb = ProgressBar(1000, name = "Learning!", timecount = False, completionprediction=True, colored=False)
pb.begin()
#Let the training begin!
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict = {x: batch[0], y_:batch[1]})
    pb.inc()
pb.end()
# A thousand repititions, like in the gym!


"""
Was the training good for the model? how so?

"""
#Does the precition match the truth?
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#How well does it match it?
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict = {x: mnist.test.images, y_:mnist.test.labels}))

# Outputs:
# 0.9171
