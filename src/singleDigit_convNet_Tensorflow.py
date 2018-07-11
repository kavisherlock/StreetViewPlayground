import os
import time
import math
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# Open the file as readonly
h5f = h5py.File('../data/SVHN_single_grey.h5', 'r')
# Load the training, test and validation set
train_dataset = h5f['X_train'][:]
train_labels = h5f['y_train'][:]
validation_dataset = h5f['X_val'][:]
validation_labels = h5f['y_val'][:]
test_dataset = h5f['X_test'][:]
test_labels = h5f['y_test'][:]
# Close this file
h5f.close()

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', validation_dataset.shape, validation_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# We know that SVHN images have 32 pixels in each dimension
image_size = train_dataset.shape[1]
# Greyscale images only have 1 color channel
num_channels = train_dataset.shape[-1]
# Number of classes, one class for each of 10 digits
num_labels = train_labels.shape[1]

# Calculate the mean on the training data
train_mean = np.mean(train_dataset, axis=0)
# Calculate the std on the training data
train_std = np.std(train_dataset, axis=0)
# Subtract it equally from all splits
train_dataset = (train_dataset - train_mean) / train_std
validation_dataset = (validation_dataset - train_mean) / train_std
test_dataset = (test_dataset - train_mean) / train_std


def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    """ Plot nrows * ncols images from images and annotate the images"""
    # Initialize the subplotgrid
    fig, axes = plt.subplots(nrows, ncols)
    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows * ncols)
    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat):
        # Predictions are not passed
        if cls_pred is None:
            title = "True: {0}".format(np.argmax(cls_true[i]))
        # When predictions are passed, display labels + predictions
        else:
            title = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), cls_pred[i])
        # Display the image
        ax.imshow(images[i, :, :, 0], cmap='binary')
        # Annotate the image
        ax.set_title(title)
        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# Plot 2 rows with 9 images each from the training set
plot_images(train_dataset, 2, 9, train_labels);

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 100 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

