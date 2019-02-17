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


def conv_weight_variable(layer_name, shape):
    """ Retrieve an existing variable with the given layer name"""
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def fc_weight_variable(layer_name, shape):
    """ Retrieve an existing variable with the given layer name"""
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    """ Creates a new bias variable"""
    return tf.Variable(tf.constant(0.0, shape=shape))


def conv_layer(layer_input,  # The previous layer
               layer_name,  # Layer name
               num_input_channels,  # Num. channels in prev. layer
               filter_size,  # Width and height of each filter
               num_filters,  # Number of filters
               pooling=True):  # Use 2x2 max-pooling

    # Shape of the filter-weights for the convolution
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new filters with the given shape
    weights = conv_weight_variable(layer_name, shape=shape)

    # Create new biases, one for each filter
    biases = bias_variable(shape=[num_filters])

    # Create the TensorFlow operation for convolution
    layer = tf.nn.conv2d(input=layer_input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')  # with zero padding

    # Add the biases to the results of the convolution
    layer += biases

    # Rectified Linear Unit (RELU)
    layer = tf.nn.relu(layer)

    # Down-sample the image resolution?
    if pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Return the resulting layer and the filter-weights
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Return the flattened layer and the number of features.
    return layer_flat, num_features


def fc_layer(layer_input,  # The previous layer
             layer_name,  # The layer name
             num_inputs,  # Num. inputs from prev. layer
             num_outputs,  # Num. outputs
             relu=True):   # Use RELU?

    # Create new weights and biases.
    weights = fc_weight_variable(layer_name, shape=[num_inputs, num_outputs])
    biases = bias_variable(shape=[num_outputs])

    # Calculate the layer activation
    layer = tf.matmul(layer_input, weights) + biases

    # Use ReLU?
    if relu:
        layer = tf.nn.relu(layer)

    return layer


# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 256            # Number of neurons in fully-connected layer.

x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels), name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
keep_prob = tf.placeholder(tf.float32)
conv_1, w_c1 = conv_layer(layer_input=x,
                          layer_name="conv_1",
                          num_input_channels=num_channels,
                          filter_size=filter_size1,
                          num_filters=num_filters1,
                          pooling=True)
dropout = tf.nn.dropout(conv_1, keep_prob)

conv_2, w_c2 = conv_layer(layer_input=conv_1,
                          layer_name="conv_2",
                          num_input_channels=num_filters1,
                          filter_size=filter_size2,
                          num_filters=num_filters2,
                          pooling=True)
dropout = tf.nn.dropout(conv_2, keep_prob)

layer_flat, num_features = flatten_layer(dropout)

fc_1 = fc_layer(layer_input=layer_flat,
                layer_name="fc_1",
                num_inputs=num_features,
                num_outputs=fc_size,
                relu=True)

fc_2 = fc_layer(layer_input=fc_1,
                layer_name="fc_2",
                num_inputs=fc_size,
                num_outputs=num_labels,
                relu=False)

y_pred = tf.nn.softmax(fc_2)
# The class-number is the index of the largest element.
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Calcualte the cross-entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=y_true)
# Take the average of the cross-entropy for all the image classifications.
cost = tf.reduce_mean(cross_entropy)
# Global step is required to compute the decayed learning rate
global_step = tf.Variable(0)
# Apply exponential decay to the learning rate
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96, staircase=True)
# Construct a new Adam optimizer
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)
# Predicted class equals the true class of each image?
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# Cast predictions to float and calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.initialize_all_variables())
saver = tf.train.Saver()

save_dir = 'checkpoints/'

# Create directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'svhn_single_greyscale')
#saver.restore(sess=session, save_path=save_path)
# Number of training samples in each iteration
batch_size = 64

# Keep probability in dropout layer
dropout = 0.5

total_iterations = 0


def optimize(num_iterations, display_step):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for step in range(num_iterations):

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict_train = {x: batch_data, y_true: batch_labels, keep_prob: dropout}

        # Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every display_step
        if step % display_step == 0:
            # Calculate the accuracy on the training-set.
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
            print("Minibatch accuracy at step %d: %.4f" % (step, batch_acc))

            # Calculate the accuracy on the validation-set
            validation_acc = session.run(accuracy, {x: validation_dataset, y_true: validation_labels, keep_prob: 1.0})
            print("Validation accuracy: %.4f" % validation_acc)

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Calculate the accuracy on the test-set
    test_accuracy = session.run(accuracy, {x: test_dataset, y_true: test_labels, keep_prob: 1.0})
    print("Test accuracy: %.4f" % test_accuracy)


optimize(num_iterations=1000, display_step=200)
# optimize(num_iterations=50000, display_step=1000)
saver.save(sess=session, save_path=save_path)

# Generate predictions for the testset
test_pred = session.run(y_pred_cls, {x: test_dataset, y_true: test_labels, keep_prob: 1.0})

# Find the incorrectly classified examples
incorrect = test_pred != np.argmax(test_labels, axis=1)

# Select the incorrectly classified examples
images = test_dataset[incorrect]
cls_true = test_labels[incorrect]
cls_pred = test_pred[incorrect]

# Plot the mis-classified examples
# plot_images(images, 3, 6, cls_true, cls_pred);

# Find the incorrectly classified examples
correct = np.invert(incorrect)

# Select the correctly classified examples
images = test_dataset[correct]
cls_true = test_labels[correct]
cls_pred = test_pred[correct]

# Plot the mis-classified examples
# plot_images(images, 3, 6, cls_true, cls_pred);

