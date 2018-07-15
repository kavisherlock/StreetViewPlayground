from __future__ import print_function
import os
import tarfile
from urllib.request import urlretrieve
from six.moves import cPickle as pickle
import pandas as pd
from PIL import Image

from helpers import digit_struct_mat_to_csv

def maybe_download(base_url, dest_folder, filename, force_download=False):
    """Download a file if not present"""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filepath = os.path.join(dest_folder, filename)
    if force_download or not os.path.exists(filepath):
        print('Attempting to download:', filename)
        filepath, _ = urlretrieve(base_url + filename, filepath)
        print('\nDownload Complete!')
    else:
        print(filename, 'already exists. Skipping download.')
    return filename


def maybe_extract(filename, dest_folder, force=False):
    extraction_dir = filename.split(".")[0]
    if force or not os.path.isdir(extraction_dir):
        print('Extracting', filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(dest_folder)
        tar.close()
        print(filename + " extracted to " + extraction_dir)
    else:
        print("Folder " + extraction_dir + " already exists. Skipping extraction.")


def save_as_pickle(data, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print(filename + " pickled!")
    except Exception as e:
        print('Unable to save data to', filename, ':', e)


def load_pickle(file):
    with open(file, 'rb') as pickle_file:
        return pickle.load(pickle_file)


# using https://github.com/sarahrn/Py-Gsvhn-DigitStruct-Reader to convert .mat files to .csv
def maybe_mat_to_csv(source_mat_file, dest_csv_file, force=False):
    if not force and os.path.isfile(dest_csv_file):
        print(source_mat_file + " already converted to " + dest_csv_file)
    else:
        print("Converting " + source_mat_file)
        digit_struct_mat_to_csv.writeToCsvFile(source_mat_file, dest_csv_file)
        print(source_mat_file + " converted to " + dest_csv_file)


def get_image_size(filepath):
    """Returns the image size in pixels given as a 2-tuple (width, height)"""
    if not os.path.exists(filepath):
        print(filepath + " does not exist")
        return
    image = Image.open(filepath)
    return image.size


def get_image_sizes(folder):
    """Returns a DataFrame with the file name and size of all images contained in a folder"""
    image_sizes = []

    # Get all .png images contained in the folder
    images = [img for img in os.listdir(folder) if img.endswith('.png')]

    # Get image size of every individual image
    for image in images:
        w, h = get_image_size(folder + '/' + image)
        image_size = {'FileName': image, 'image_width': w, 'image_height': h}
        image_sizes.append(image_size)

    # Return results as a pandas DataFrame
    return pd.DataFrame(image_sizes)
