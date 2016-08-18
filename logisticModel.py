"""Logistic model."""

import tensorflow as tf
import numpy as np
import data

# Lets define our accuracy
def accuracy(predictions, real_outputs):
    return np.sqrt( np.mean( np.square(predictions - real_outputs)))

def accuracy2(predictions, real_outputs):
    return np.sqrt( np.mean( np.square( np.log(predictions+1) - np.log(real_outputs+1))))

nExamples = 5000000
train_size = 60000
valid_size = 10000
test_size = 20000

# How many steps to do to train the model
num_steps = 2000

train_dataset, train_output, valid_dataset, valid_output, test_dataset, test_output = data.getRandomDatasets(nExamples, train_size, valid_size, test_size)

# How many features has our charasteristic vector
features = train_dataset.shape[1]

# Hyper-parameter for the GD
alpha = 0.00001

graph = tf.Graph()
with graph.as_default():

    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(train_dataset)
    tf_train_output = tf.constant(train_output)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random valued following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(
      tf.truncated_normal([features, 1]))
    biases = tf.Variable(tf.zeros([1]))

    # Training computation.

    # We only want positive ooutputs, we'll use RELU for that
    output = tf.nn.relu( tf.matmul(tf_train_dataset, weights) + biases )

    loss = tf.nn.l2_loss( output - tf_train_output)

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = output
    valid_prediction = tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases)

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    tf.initialize_all_variables().run()
    print('Initialized')

    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])

        if (step % 200 == 0):
            print('------------------')
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy:', accuracy2(predictions, train_output))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            print('Validation accuracy:', accuracy2(valid_prediction.eval(), valid_output))
    print('Test accuracy:', accuracy2(test_prediction.eval(), test_output))
