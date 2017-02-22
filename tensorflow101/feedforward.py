import tensorflow as tf
import numpy as np

import input_data
import mnist

data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

images_placeholder = tf.placeholder(tf.float32, shape = (batch_size,
                                                        mnist.IMAGE_PIXELS))

labels_placeholder = tf.placeholder(tf.int32, shape = (batch_size))
