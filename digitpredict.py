import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
#None means that the dimension can be of any lenght
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
"""
 Hacky stuff, W * x is written as x * W in tf.matmul
 This is to deal with x being a 2D tensor
 Tracking how bad de model is by using the cost/loss function
 Enter, cross-entropy http://colah.github.io/posts/2015-09-Visual-Information/
"""
y_ = tf.placeholder(tf.float32, [None, 10])

# Implement cross-entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
# Command tensorflow to minimize cross_entropy with 0.5 steps in a given direction
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#Interactive session
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

# Let the TRAIN begin
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

model_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
print("Accuracy: %s %%"%(model_accuracy * 100))
"""
 Outputs
 Accuracy: 91.83%
 Congrats, you've done your first Machine Learning Algorithm
 With bad results!

"""
