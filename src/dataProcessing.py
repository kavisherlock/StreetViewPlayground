import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from helpers import data_helper
from IPython.display import display
from PIL import Image, ImageDraw
from scipy.ndimage import imread
from scipy.misc import imresize


def display_bbox(image_folder, image_name, bbox):
    """ Helper function to display a single image and bounding box"""
    image_path = image_folder + '/' + image_name
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle([bbox.x0, bbox.y0, bbox.x1, bbox.y1], outline='blue')
    image.show()
    return image


def refine_data_frame(digit_info, folder):
    """Returns a DataFrame with the more detail than available in digitStruct.mat"""
    # Rename Top and Left to y0 and x0, and adding Width and Height to get x1 and y1
    digit_info.rename(columns={'Left': 'x0', 'Top': 'y0', 'DigitLabel': 'label'}, inplace=True)
    digit_info['x1'] = digit_info['x0'] + digit_info['Width']
    digit_info['y1'] = digit_info['y0'] + digit_info['Height']

    # Creating a bounding box around the entire number instead of just the digit
    aggregate = {'x0': 'min',
                 'y0': 'min',
                 'x1': 'max',
                 'y1': 'max',
                 'label': {
                    'labels': lambda x: list(x),
                    'num_digits': 'count'}}
    digit_info = digit_info.groupby('FileName').agg(aggregate).reset_index()
    digit_info.columns = [x[0] if i < 5 else x[1] for i, x in enumerate(digit_info.columns.values)]

    # Expand the bounding boxes by 30%
    digit_info['x_increase'] = ((digit_info['x1'] - digit_info['x0']) * 0.3) / 2.
    digit_info['y_increase'] = ((digit_info['y1'] - digit_info['y0']) * 0.3) / 2.
    digit_info['x0'] = (digit_info['x0'] - digit_info['x_increase']).astype('int')
    digit_info['y0'] = (digit_info['y0'] - digit_info['y_increase']).astype('int')
    digit_info['x1'] = (digit_info['x1'] + digit_info['x_increase']).astype('int')
    digit_info['y1'] = (digit_info['y1'] + digit_info['y_increase']).astype('int')

    # Merge image widths and heights to data
    image_sizes = data_helper.get_image_sizes(folder)
    digit_info = pd.merge(digit_info, image_sizes, on='FileName', how='inner')
    del image_sizes

    # Make sure that expanded bounding boxes are contained by the image
    digit_info.loc[digit_info['x0'] < 0, 'x0'] = 2
    digit_info.loc[digit_info['y0'] < 0, 'y0'] = 2
    digit_info.loc[digit_info['x1'] > digit_info['image_width'], 'x1'] = digit_info['image_width'] - 2
    digit_info.loc[digit_info['y1'] > digit_info['image_height'], 'y1'] = digit_info['image_height'] - 2
    # image_name = '12112.png'
    # bbox = digit_info[digit_info['FileName'] == image_name]
    # display_bbox('../data/train/', image_name, bbox)

    # Keep only images with less than 6 digits
    digit_info = digit_info[digit_info['num_digits'] < 6]

    return digit_info


def csv_to_data_frame(csv_folder, force=False):
    """Converts csv to dataframe and refines it"""
    if force or not os.path.exists(os.path.join(csv_folder, 'digitStruct.dataframe')):
        print('Creating ' + csv_folder + '/digitStruct.dataframe')
        data_frame = pd.read_csv(os.path.join(csv_folder, 'digitStruct.csv'))
        data_frame = refine_data_frame(data_frame, csv_folder)
        data_helper.save_as_pickle(data_frame, os.path.join(csv_folder, 'digitStruct.dataframe'))
        return data_frame
    else:
        print(csv_folder + '/digitStruct.dataframe already exists')
        return data_helper.load_pickle(os.path.join(csv_folder, 'digitStruct.dataframe'))


def crop_and_resize(digit_folder, image, img_size):
    """Crop and resize an image"""
    try:
        image_data = imread(digit_folder + '/' + image['FileName'])
        crop = image_data[image['y0']:image['y1'], image['x0']:image['x1'], :]
        return imresize(crop, img_size)
    except ValueError:
        print('Value Error!!!!!!!!!!!')
        print(image)


def create_dataset(digit_folder, digit_info, img_size):
    """ Helper function for converting images into a numpy array"""
    print('Creating ' + digit_folder + ' dataset')
    # Initialize the numpy arrays (0's are stored as 10's)
    X = np.zeros(shape=(digit_info.shape[0], img_size[0], img_size[0], 3), dtype='uint8')
    y = np.full((digit_info.shape[0], 5), 10, dtype=int)

    # Iterate over all images in the pandas dataframe (slow!)
    for i, (index, image) in enumerate(digit_info.iterrows()):
        # Get the image data
        X[i] = crop_and_resize(digit_folder, image, img_size)

        # Get the label list as an array
        labels = np.array((image['labels']))

        # Store 0's as 0 (not 10)
        labels[labels == 10] = 0

        # Embed labels into label array
        y[i, 0:labels.shape[0]] = labels

    # Return data and labels
    return X, y


def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    """ Helper function for plotting nrows * ncols images"""
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2*nrows))
    for i, ax in enumerate(axes.flat):
        # Pretty string with actual label
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)
        if cls_pred is None:
            title = "True: {0}".format(true_number)
        else:
            # Pretty string with predicted label
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "True: {0}, Pred: {1}".format(true_number, pred_number)
        ax.imshow(images[i])
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()


def random_sample(N, K):
    """Return a boolean mask of size N with K selections"""
    mask = np.array([True]*K + [False]*(N-K))
    np.random.shuffle(mask)
    return mask


force_download = False
download_url = 'http://ufldl.stanford.edu/housenumbers/'
data_folder = '../data/'
train_filename = 'train.tar.gz'
test_filename = 'test.tar.gz'

data_helper.maybe_download(download_url, data_folder, train_filename)
data_helper.maybe_download(download_url, data_folder, test_filename)

os.chdir('../data')
data_helper.maybe_extract(train_filename, data_folder)
data_helper.maybe_extract(test_filename, data_folder)

data_helper.maybe_mat_to_csv(os.path.join('train', 'digitStruct.mat'), os.path.join('train', 'digitStruct.csv'))
data_helper.maybe_mat_to_csv(os.path.join('test', 'digitStruct.mat'), os.path.join('test', 'digitStruct.csv'))

training_digit_info = csv_to_data_frame('train')
# display(training_digit_info.head())
test_digit_info = csv_to_data_frame('test')
# display(test_digit_info.head())

# image = '100.png'
# bbox = training_digit_info[training_digit_info.FileName == image]
# display_bbox('test', image, bbox)

if not os.path.exists('SVHN_multi.h5'):
    image_size = (32, 32)
    X_train, y_train = create_dataset('train', training_digit_info, image_size)
    X_test, y_test = create_dataset('test', test_digit_info, image_size)

    del training_digit_info
    del test_digit_info

    # Create valdidation from the sampled data
    sample1 = random_sample(X_train.shape[0], 5000)
    X_val = X_train[sample1]
    y_val = y_train[sample1]
    X_train = X_train[~sample1]
    y_train = y_train[~sample1]

    # Store the datasets
    h5f = h5py.File('SVHN_multi.h5', 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_val', data=X_val)
    h5f.create_dataset('y_val', data=y_val)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.close()
else:
    # Open the HDF5 file containing the datasets and sxtract the datasets
    h5f = h5py.File('SVHN_multi.h5', 'r')
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    h5f.close()

print("Training", X_train.shape, y_train.shape)
print('Validation', X_val.shape, y_val.shape)
print("Test", X_test.shape, y_test.shape)
# plot_images(X_train, 3, 6, y_train)
# plot_images(X_test, 3, 6, y_test)


def rgb2gray(images):
    """Convert images from rbg to grayscale
    """
    greyscale = np.dot(images, [0.2989, 0.5870, 0.1140])
    return np.expand_dims(greyscale, axis=3)


if not os.path.exists('SVHN_multi_grey.h5'):
    # Transform the images to greyscale
    X_train = rgb2gray(X_train).astype(np.float32)
    X_test = rgb2gray(X_test).astype(np.float32)
    X_val = rgb2gray(X_val).astype(np.float32)

    # Store the datasets
    h5f = h5py.File('SVHN_multi_grey.h5', 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_val', data=X_val)
    h5f.create_dataset('y_val', data=y_val)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.close()

os.chdir('../src')
