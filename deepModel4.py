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

nExamples = 100000
train_size = 50000
valid_size = 10000
test_size = 20000

# How many steps to do to train the model
num_steps = 1000

train_dataset, train_output, valid_dataset, valid_output, test_dataset, test_output = data.getRandomDatasets(nExamples, train_size, valid_size, test_size)

# How many features has our charasteristic vector
features = train_dataset.shape[1]

# Hyper-parameters
hidden_nodes1 = 64
hidden_nodes2 = 32
hidden_nodes3 = 16
hidden_nodes4 = 8
# GD
alpha1 = 0.00001
alpha2 = 0.0000001
# Regularization
beta1 = 0.0001
beta2 = 0.0001
beta3 = 0.0001
beta4 = 0.0001
beta5 = 0.0001

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
    weights3 = tf.Variable( tf.truncated_normal([hidden_nodes2, hidden_nodes3]))
    biases3 = tf.Variable(tf.zeros([hidden_nodes3]))

    # Hidden layer 3
    weights4 = tf.Variable( tf.truncated_normal([hidden_nodes3, hidden_nodes4]))
    biases4 = tf.Variable( tf.zeros([hidden_nodes4]))

    # Hidden layer 4
    weights5 = tf.Variable( tf.truncated_normal( [hidden_nodes4, 1]))
    biases5 = tf.Variable( tf.zeros([1]))

    #----------------------
    # Training computation.
    #----------------------

    # Outputs after the first layer:
    # We are using tanh for the activation function, thats the non linearity in our model
    outputs1 = tf.nn.tanh( tf.matmul(tf_train_dataset, weights1) + biases1 )

    # Output after the second layer:
    outputs2 = tf.nn.tanh( tf.matmul(outputs1, weights2) + biases2)

    # Output after the third layer:
    outputs3 = tf.nn.tanh( tf.matmul(outputs2, weights3) + biases3)

    # Output after the fourth layer:
    outputs4 = tf.nn.tanh( tf.matmul(outputs3, weights4) + biases4)

    # Output of the model
    output = tf.nn.relu( tf.matmul(outputs4, weights5) + biases5)


    #-------
    # Loss
    #-------
    loss = tf.nn.l2_loss( tf.log(output+1) - tf.log(tf_train_outputs+1) ) + beta1*tf.nn.l2_loss(weights1) + beta2*tf.nn.l2_loss(weights2) + beta3*tf.nn.l2_loss(weights3) + beta4*tf.nn.l2_loss(weights4) + beta5*tf.nn.l2_loss(weights5)

    #-----------
    # Optimizer
    #-----------

    tf_alpha = tf.placeholder('float32')
    optimizer = tf.train.GradientDescentOptimizer(tf_alpha).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = output

    # Valid prediction:
    valid_outputs1 = tf.nn.tanh( tf.matmul(tf_valid_dataset, weights1) + biases1 )
    valid_outputs2 = tf.nn.tanh( tf.matmul(valid_outputs1, weights2) + biases2)
    valid_outputs3 = tf.nn.tanh( tf.matmul(valid_outputs2, weights3) + biases3)
    valid_outputs4 = tf.nn.tanh( tf.matmul(valid_outputs3, weights4) + biases4)
    valid_prediction = tf.nn.relu( tf.matmul(valid_outputs4, weights5) + biases5)

    # Test prediction:
    test_outputs1 = tf.nn.tanh( tf.matmul(tf_test_dataset, weights1) + biases1 )
    test_outputs2 = tf.nn.tanh( tf.matmul(test_outputs1, weights2) + biases2)
    test_outputs3 = tf.nn.tanh( tf.matmul(test_outputs2, weights3) + biases3)
    test_outputs4 = tf.nn.tanh( tf.matmul(test_outputs3, weights4) + biases4)
    test_prediction = tf.nn.relu( tf.matmul(test_outputs4, weights5) + biases5)

# Run the model

with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()
    print("Initialized")
    begin = time.time()

    for step in range(num_steps):

        alpha_feed = {tf_alpha : alpha1 + step*alpha2}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=alpha_feed)

        if (step % 10 == 0):
            print('-----------')
            print("Train loss at step %d: %f" % (step, l))
            print('Train accuracy:', accuracy2(predictions, train_output))
            print('Validation accuracy:', accuracy2(valid_prediction.eval(), valid_output))

            print('Time (s):', time.time()-begin)
            begin = time.time()

    print('Test accuracy:', accuracy2(test_prediction.eval(), test_output))

    # Let's print some results: predictions vs real
    print('-----------')
    print('Prediction vs Real:')
    print(np.concatenate((test_prediction.eval(), test_output), axis=1)[0:100])
