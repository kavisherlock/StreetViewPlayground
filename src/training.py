import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import time
import os
from datetime import timedelta
from sklearn.utils import shuffle

from convNet_model import predict


def subtract_mean(a):
    """ Helper function for subtracting the mean of every image"""
    for i in range(a.shape[0]):
        a[i] -= a[i].mean()
    return a


def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    # Initialize figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2 * nrows))
    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows * ncols)
    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat):
        # Pretty string with actual number
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)
        if cls_pred is None:
            title = "True: {0}".format(true_number)
        else:
            # Pretty string with predicted number
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "True: {0}, Pred: {1}".format(true_number, pred_number)
        ax.imshow(images[i, :, :, 0], cmap='binary')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# Open the HDF5 file containing the datasets
h5f = h5py.File('../data/SVHN_multi_grey.h5', 'r')

# Extract the datasets
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]

# Close the file
h5f.close()

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

# Get the data dimensions
_, img_height, img_width, num_channels = X_train.shape

# ... and label information
num_digits, num_labels = y_train.shape[1], len(np.unique(y_train))

X_train, y_train = shuffle(X_train, y_train)

# Subtract the mean from every image
X_train = subtract_mean(X_train)
X_test = subtract_mean(X_test)
X_val = subtract_mean(X_val)

# plot_images(X_train, 2, 8, y_train)

with tf.name_scope("input"):
    # Placeholders for feeding input images
    x = tf.placeholder(tf.float32, shape=(None, img_height, img_width, num_channels), name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, num_digits], name='y_')

with tf.name_scope("dropout"):
    # Dropout rate applied to the input layer
    p_keep_1 = tf.placeholder(tf.float32)
    tf.summary.scalar('input_keep_probability', p_keep_1)

    # Dropout rate applied after the pooling layers
    p_keep_2 = tf.placeholder(tf.float32)
    tf.summary.scalar('conv_keep_probability', p_keep_2)

    # Dropout rate using between the fully-connected layers
    p_keep_3 = tf.placeholder(tf.float32)
    tf.summary.scalar('fc_keep_probability', p_keep_3)

dropout_rates = [p_keep_1, p_keep_2, p_keep_3]

logits, y_pred_cls = predict(x, dropout_rates, num_channels, num_labels)

with tf.name_scope('loss'):
    # Calculate the loss for each individual digit in the sequence
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[0], labels=y_[:, 0]))
    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[1], labels=y_[:, 1]))
    loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[2], labels=y_[:, 2]))
    loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[3], labels=y_[:, 3]))
    loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[4], labels=y_[:, 4]))

    # Calculate the total loss for all predictions
    loss = loss1 + loss2 + loss3 + loss4 + loss5
    tf.summary.scalar('loss', loss)

with tf.name_scope('optimizer'):
    # Global step is required to compute the initializing variablesdecayed learning rate
    global_step = tf.Variable(0, trainable=False)

    # Apply exponential decay to the learning rate
    learning_rate = tf.train.exponential_decay(1e-3, global_step, 7500, 0.5, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    # Construct a new Adam optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.name_scope("accuracy"):
    # Predicted class equals the true class of each image?
    correct_prediction = tf.reduce_min(tf.cast(tf.equal(y_pred_cls, y_), tf.float32), 1)

    # Cast predictions to float and calculate the mean
    accuracy = tf.reduce_mean(correct_prediction) * 100.0

    # Add scalar summary for accuracy tensor
    tf.summary.scalar('accuracy', accuracy)


session = tf.Session()
saver = tf.train.Saver()

save_path = os.path.join('checkpoints/', 'svhn_multi_v5.ckpt')

# Use TensorFlow to find the latest checkpoint - if any
try:
    print("Restoring last checkpoint from " + save_path +  " ...")

    # Finds the filename of latest saved checkpoint file
    last_chk_path = tf.train.latest_checkpoint('checkpoints/')

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)

# If the above failed - initialize all the variables
except Exception as e:
    print("Failed to restore checkpoint - initializing variables:", e)
    session.run(tf.global_variables_initializer())

LOG_DIR = 'logs/svhn_multi_v5/'

# Merge all the summaries and write them out to /logs/svhn_multi
merged = tf.summary.merge_all()

# Pass the graph to the writer to display it in TensorBoard
train_writer = tf.summary.FileWriter(LOG_DIR + '/train', session.graph)
validation_writer = tf.summary.FileWriter(LOG_DIR + '/validation')

# If you run out of memory - switch to a smaller batch size
batch_size = 128
# Dropout applied to the input layer
d1 = 0.9
# Dropout applied between the conv layers
d2 = 0.75
# Dropout applied to the fully-connected layers
d3 = 0.5


def feed_dict(step=0):
    """ Make a TensorFlow feed_dict mapping data onto the placeholders
    """
    # Calculate the offset
    offset = (step * batch_size) % (y_train.shape[0] - batch_size)

    # Get the batch data
    xs, ys = X_train[offset:offset + batch_size], y_train[offset:offset + batch_size]

    return {x: xs, y_: ys, p_keep_1: d1, p_keep_2: d2, p_keep_3: d3}


def evaluate_batch(test, batch_size):
    """ Evaluate in batches to avoid out-of-memory issues
    """
    # Store the cumulative accuracy over all batches
    cumulative_accuracy = 0.0

    # Get the number of images
    n_images = y_test.shape[0] if test else y_val.shape[0]

    # Numer of batches needed to evaluate all images
    n_batches = n_images // batch_size + 1

    # Iterate over all the batches
    for i in range(n_batches):

        # Calculate the offset
        offset = i * batch_size

        if test:
            # Get the batch from the test set
            xs, ys = X_test[offset:offset + batch_size], y_test[offset:offset + batch_size]
        else:
            # Get batch from the validation set
            xs, ys = X_val[offset:offset + batch_size], y_val[offset:offset + batch_size]

        cumulative_accuracy += session.run(accuracy,
                                           {x: xs, y_: ys, p_keep_1: 1., p_keep_2: 1., p_keep_3: 1.})

    # Return the average accuracy over all batches
    return cumulative_accuracy / (0.0 + n_batches)


def optimize(num_iterations, display_step):
    # Start-time used for printing time-usage
    start_time = time.time()

    for step in range(num_iterations):

        # Run the optimizer using this batch of training data.
        summary, i, _ = session.run([merged, global_step, optimizer], feed_dict(step))
        train_writer.add_summary(summary, i)

        # Print the status every display_step iteration and last
        if (i % display_step == 0) or (step == num_iterations - 1):
            # Calculate the minibatch accuracy
            batch_acc = session.run(accuracy, feed_dict=feed_dict(step))
            print("Minibatch accuracy at step %d: %.4f" % (i, batch_acc))

            # Calculate the accuracy on the validation-set
            # valid_acc = evaluate_batch(test=False, batch_size=512)
            # print("Validation accuracy at step %s: %.4f" % (i, valid_acc))

    # Total training time
    run_time = time.time() - start_time
    print("\nTime usage: " + str(timedelta(seconds=int(round(run_time)))))

    # Calculate and display the testset accuracy
    test_acc = evaluate_batch(test=True, batch_size=512)
    print("Test accuracy: %.4f" % test_acc)

    # Save all the variables of the TensorFlow graph
    saver.save(session, save_path=save_path, global_step=global_step)
    print('Model saved in file: {}'.format(save_path))

optimize(num_iterations=200, display_step=100)

# Feed the test set with dropout disabled
feed_dict={
    x: X_test,
    y_: y_test,
    p_keep_1: 1.,
    p_keep_2: 1.,
    p_keep_3: 1.
}

# Generate predictions for the testset
test_pred = session.run(y_pred_cls, feed_dict=feed_dict)

# Display the predictions
test_pred


def calculate_accuracy(a, b):
    """ Calculating the % of similar rows in two numpy arrays
    """
    # Compare two numpy arrays row-wise
    correct = np.sum(np.all(a == b, axis=1))
    return 100.0 * (correct / (0.0 + a.shape[0]))


# For every possible sequence length
for num_digits in range(1, 6):
    # Find all images with that given sequence length
    images = np.where((y_test != 10).sum(1) == num_digits)

    # Calculate the accuracy on those images
    acc = calculate_accuracy(test_pred[images], y_test[images])

    print("%d digit accuracy %.3f %%" % (num_digits, acc))
