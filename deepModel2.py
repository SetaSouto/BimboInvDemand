"""Deep Neural Network model."""

import tensorflow as tf
import numpy as np
import time
import data

# Lets define our accuracy
def accuracy(predictions, real_outputs):
    return np.sqrt( np.mean( np.square(predictions - real_outputs)))

def accuracy2(predictions, real_outputs):
    return np.sqrt( np.mean( np.square( np.log(predictions+1) - np.log(real_outputs+1))))

nExamples = 5000000
train_size = 30000
valid_size = 10000
test_size = 20000

# How many steps to do to train the model
num_steps = 2000

train_dataset, train_output, valid_dataset, valid_output, test_dataset, test_output = data.getRandomDatasets(nExamples, train_size, valid_size, test_size)

# How many features has our charasteristic vector
features = train_dataset.shape[1]
hidden_nodes1 = 256
hidden_nodes2 = 128
alpha = 0.0000005

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.constant(train_dataset)
    # We only have one output, the prediction.
    tf_train_outputs = tf.constant(train_output)
    # The valid dataset and the test dataset remain as constants.
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    #------------
    # Variables:
    #------------

    # First layer
    weights1 = tf.Variable(
        tf.truncated_normal([features, hidden_nodes1]))
    biases1 = tf.Variable(tf.zeros([hidden_nodes1]))

    # Hidden layer 1
    weights2 = tf.Variable(
        tf.truncated_normal([hidden_nodes1, hidden_nodes2]))
    biases2 = tf.Variable(tf.zeros([hidden_nodes2]))

    # Hidden layer 2
    weights3 = tf.Variable( tf.truncated_normal( [hidden_nodes2, 1]))
    biases3 = tf.Variable( tf.zeros([1]))

    #----------------------
    # Training computation.
    #----------------------

    # Outputs after the first layer:
    # We are using tanh for the activation function, thats the non linearity in our model
    outputs1 = tf.nn.tanh( tf.matmul(tf_train_dataset, weights1) + biases1 )

    # Output after the second layer:
    outputs2 = tf.nn.tanh( tf.matmul(outputs1, weights2) + biases2)

    # Output of the model
    output = tf.nn.relu( tf.matmul(outputs2, weights3) + biases3)


    #-------
    # Loss
    #-------
    loss = tf.nn.l2_loss( output - tf_train_outputs)

    #-----------
    # Optimizer
    #-----------

    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = output
    valid_prediction = tf.nn.relu( tf.matmul(
                                        (tf.matmul( tf.nn.tanh( tf.matmul(tf_valid_dataset, weights1) + biases1 ),
                                                   weights2) + biases2),
                                        weights3) + biases3)
    test_prediction = tf.nn.relu( tf.matmul( (tf.matmul(tf.nn.tanh( tf.matmul(tf_test_dataset, weights1) + biases1 ) , weights2) + biases2),
                                              weights3) + biases3)

# Run the model

with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()
    print("Initialized")
    begin = time.time()

    for step in range(num_steps):

        _, l, predictions = session.run([optimizer, loss, train_prediction])

        if (step % 10 == 0):
            print('-----------')
            print("Train loss at step %d: %f" % (step, l))
            print('Train accuracy:', accuracy2(predictions, train_output))
            print('Validation accuracy:', accuracy2(valid_prediction.eval(), valid_output))

            print('Time (s):', time.time()-begin)
            begin = time.time()

    print('Test accuracy:', accuracy2(test_prediction.eval(), test_output))
