import tensorflow as tf
import numpy as np

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
#Outputs:
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# Notice that it doesn't output the values of the nodes
# Use a SESSION to do this

sess = tf.Session()
print (sess.run([node1, node2]))
# Outputs
# [3.0, 4.0]

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))
# Outputs
# node3:  Tensor("Add_2:0", shape=(), dtype=float32)
# sess.run(node3):  7.0

# A graph can be parameterized to accept external inputs,
# known as placeholders
# A placeholder is a promise to provide a value later

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for tf.add(a,b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2,4]}))
#Outputs
#7.5
#[ 3.  7.]
#Notice that one could assing the values within the print
#that's the whole point of the placeholder


add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))

# Outputs
# 22.5

# To make a model trainable we need to be able to modify
# the graph to get new outputs with the same input
# Variable allow us to add trainable parameters to a graph.
# They are constructed with a type and initial values

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# constants are initialized with tf.constant and their value can never change

# variables are not initialized when you call tf.variables
# to initialize ALL the variables in a TensorFlow program
# you must explicitly call a special operation as follows

init = tf.global_variables_initializer() # A handle to the Tensorflow sub-graph
sess.run(init) # Until we call sess.run the variables are uninitialized

# Since x is a placeholder in the linear_model,
# we can evaluate the model for several values of x simultaneously

print(sess.run(linear_model, {x:[1,2,3,4]}))
# Outputs
# [ 0.          0.30000001  0.60000002  0.90000004]

"""
So far we've created a model but we don't know yet how good it is.
A 'y' placeholder is needed to provide the desired values
and a loss function needs to be written
"""

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# linear_model - y creates a vector with the errors
# tf.square squares the erros and then we add them all up
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))
# Outputs
# 23.66

"""
Since the error is too high, W and b need to be fixed
W = -1 and b = -1
tf.Variable has initialized W and b to 0.3 and -0.3 but
these can be changed through tf.assign
"""

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))
# Outputs
# 0

"""
At this point the values for W and b were guessed
but the whole point of machine learning is to
find the correct parameters automatically
Enter, tf.train API

Tensor flow provides optimizers that slowly change
each variable in order to minimize the loss function

Gradient Descent is the simplest
It modifies each variable according to the magnitude of the
derivate of loss with respect to that variable

"""

optimizer = tf.train.GradientDescentOptimizer(0.01) # 0.01 is the step
train = optimizer.minimize(loss)

sess.run(init) # this removes the previous assing and W and b are wrong again

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W,b]))
# Outputs
# [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
# NOW THAT's actual machine learning, very close to -1 and 1

# Making the output fancier
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# Evaluate accuracy
curr_W, curr_b, curr_loss = sess.run([W,b,loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
# Outputs
# W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
