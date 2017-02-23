"""
tf.contrib.learn is the high-level machine learning API from tensorflow

In this example a Neural Network classifier is consturcted with contrib learn

1. Load CSVs containing Iris training/test data
2. Construct neural network classifier
3. Fit the model using the training data
4. Evaluate the accuracy of the model
5. Classify new samples

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# IMPORTING THE DATA SETS (REMEMBER SCI-KIT?)
IRIS_TRAINING = "iris_model/iris_training.csv" # You need to download These
IRIS_TEST = "iris_model/iris_test.csv"

# LOAD THEM into the tf.contri.learn.dataset.base

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TRAINING,
    target_dtype = np.int,
    features_dtype = np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TEST,
    target_dtype = np.int,
    features_dtype = np.float32)

# Specify that all featues have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = 4)]

# Build 3 layer DNN with 10, 20, 10 units
classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                            hidden_units = [10,20,10],
                                            n_classes = 3,
                                            model_dir = "/iris_model")

# Fit model
classifier.fit (x = training_set.data,
                y = training_set.target,
                steps = 2000)

# Accuracy evalution

accuracy_points = classifier.evaluate ( x = test_set.data,
                                        y = test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_points))
# Outputs
# 0.96667
"""
Notice that with one line you define the classifier
With another line you fit the model
and with another line you evaluate it

Could it be any simpler???!

Finally we classify (predict) two new flower samples
"""

new_samples = np.array(
            [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype = float)

y = list(classifier.predict(new_samples, as_iterable = True))

print('Predicitons: {}'.format(str(y)))
# Outputs:
# Predictions: [1, 2]

"""
About the flower classifier
It classifies the data of a flower based on its sepal width, sepal lenght,
petal length and petal width.

The Iris setosa has an id: 0
The iris versicolor has an id: 1
The iris virginica has an id: 2

For example, take a look at the following sample:

Sepal length: 5.1
Sepal width: 3.5
Petal length: 1.4
Petal width: 0.2
Species id: 0

The training set of iris_training has 120 samples
The test set of iris_test has 30 samples

In the last part of the code, after the "model" has been trained and
its accuracy measured against the test data (96.6%), we create a fake
set of data, with:

Sepal length: 6.4
Sepal width: 3.2
Petal length: 4.5
Petal width: 1.5
Species id: unknown

Sepal length: 5.8
Sepal width: 3.1
Petal length: 5.0
Petal width: 1.7
Species id: unknown

The idea is to get the species id using the "model".

The output suggests that the first fake flower is Iris versicolor
and that the second fake flower is Iris virginica.

AWESOME!!!!!
"""
