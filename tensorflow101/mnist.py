import math
import tensorflow as tf

# 0 through 9 digits, configure a number of classes equal to 10
NUM_CLASSES = 10

# The MNIST images are 28x28 pixels
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):
    """
    Build the MNIST model up to where it may

    Args:
        Images: Images placeholder, from inputs()
        hidden1_units: Size of the first hidden layer
        hidden2_units: Size of the second hidden layer

    Returns:
        softmax_linear: Output tensor with the computed goals

    """

    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name = 'weights')

        biases = tf.Variable(tf.zeros([hidden1_units]),
                                name= 'biases')

        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name = 'weights')

        biases = tf.Variable(tf.zeros([hidden2_units]),
                                name= 'biases')

        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    #Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev = 1.0 / math.sqrt(float(hidden2_units))),
            name = 'weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                            name = 'biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss(logits, labels):
    """
    Calculates the loss from the logits and labels.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES]
        labels: Labels tensor, int32 - [batch_size]

    Returns:
        Loss: loss tensor of type float
    """

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = labels, logits = logits, name = "xentropy")

    return tf.reduce_mean(cross_entropy, name = 'xentropy_mean')

def training(loss, learning_rate):
        """
        Sets up the training Ops.

        Creates a summarizer to trach the loss over time in TensorBoard

        Creates an optimized and applies the gradients to all trainable variables

        The Op returned by this function is what must be passed to the
        'sess.run()' call to cause the model to train.

        Args:
            loss: Loss tensor, from loss()
            learning_rate: The learning rate to use for gradient descent

        Returns:
            train_op: The Op for training
        """

    # Add a scalar summary for the snapshot loss
        tf.summary.scalar('loss', loss)

    # Create the gradient descent optimizer with the given learning read_data_sets
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Create a variable to track the global step
        global_step = tf.Variable(0, name = 'global_step', trainable = False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step
        train_op = optimizer.minimize(loss, global_step = global_step)
        return train_op

def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES]
        labels: Labels tensor, int32 - [batch_size], with values in
            range [0, NUM_CLASSES]
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly
    """
    # For a classifier model, in this example,
    # in_top_k Op is ued.
    # It returns a bool tensor with shape [batch_size]
    # that is true for the examples where the label is in
    # the top k, here k = 1
    # of all logits for that examples
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries
    return tf.reduce_sum(tf.cast(correct, tf.int32))

print("Successfully imported MNIST")
