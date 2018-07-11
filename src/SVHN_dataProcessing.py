import os
import sys
import tarfile
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat

import urllib.request

force_download = False
train_filename = '../data/train.tar.gz'
test_filename = '../data/test.tar.gz'

if force_download or not os.path.exists(train_filename):
    print('Downloading training data')
    urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train.tar.gz', train_filename)
    print('Downloaded training data')
if force_download or not os.path.exists(test_filename):
    print('Downloading test data')
    urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test.tar.gz', test_filename)
    print('Downloaded test data')


def extract_tarball(filename, force=False):
    """ Helper function for extracting tarball files"""
    # Drop the file extension
    root = filename.split('.')[0]
    # If file is already extracted - return
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
        return
    # If file is a tarball file - extract it
    if filename.endswith('tar.gz'):
        print('Extracting %s ...' % filename)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall()
        tar.close()
        print('Extracted %s' % filename)


ls_data = [f for f in os.listdir('../data') if 'tar.gz' in f]
os.chdir('../data')
extract_tarball(ls_data[0])
extract_tarball(ls_data[1])
os.chdir('../src')

# TODO: figure out digitStruct.mat. Look at bookmarks and the https://github.com/thomalm/svhn-multi-digit repo part 5 and 6
# TODO: create the app that allows user to create bounding box and then identifies the digit (single digit for now)
# TODO:  Deadline: before the NY trip!!