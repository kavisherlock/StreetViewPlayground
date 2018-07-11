import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import h5py

PLOT_STUFF = False


def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']


def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images"""
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])


X_train, y_train = load_data('../data/train_32x32.mat')
X_test, y_test = load_data('../data/test_32x32.mat')
# Transpose the image arrays
X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
print('Training', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
num_images = X_train.shape[0] + X_test.shape[0]
print('Total Number of Images', num_images)
print('\n')


if PLOT_STUFF:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='row')
    fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)

    # Plot some training set images
    plot_images(X_train, y_train, 2, 8)

    ax1.hist(y_train, bins=10)
    ax1.set_title('Training set')
    ax1.set_xlim(1, 10)

    ax2.hist(y_test, color='g', bins=10)
    ax2.set_title('Test set')

    fig.tight_layout()

    plt.show()

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

# Create a validation set
training_samples = []
# For every label in the dataset
for label in np.unique(y_train):
    # Get the index of all images with a specific label
    images = np.where(y_train == label)[0]
    # Draw a random sample from the images
    random_sample = np.random.choice(images, size=1000, replace=False)
    # Add the random sample to our subsample list
    training_samples += random_sample.tolist()
X_val, y_val = np.copy(X_train[training_samples]), np.copy(y_train[training_samples])
# Remove the samples to avoid duplicates
X_train = np.delete(X_train, training_samples, axis=0)
y_train = np.delete(y_train, training_samples, axis=0)
print('Training', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
print('Validation', X_val.shape, y_val.shape)
print('\n')

# Fit the OneHotEncoder
enc = OneHotEncoder().fit(y_train.reshape(-1, 1))
# Transform the label values to a one-hot-encoding scheme
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
print('After One-Hot Encoding')
print('Training', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
print('Validation', X_val.shape, y_val.shape)
print('\n')

# Create file
h5f = h5py.File('../data/SVHN_single.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data=X_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data=X_val)
h5f.create_dataset('y_val', data=y_val)

# Close the file
h5f.close()


def rgb2gray(images):
    """Convert images from rbg to grayscale"""
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3)


# Transform the images to greyscale
train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)
valid_greyscale = rgb2gray(X_val).astype(np.float32)

h5f = h5py.File('../data/SVHN_single_grey.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data=train_greyscale)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=test_greyscale)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data=valid_greyscale)
h5f.create_dataset('y_val', data=y_val)

# Close the file
h5f.close()