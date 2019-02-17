import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread

from convNet_model import predict

def rgb2gray(images):
    """Convert images from rbg to grayscale
    """
    greyscale = np.dot(images, [0.2989, 0.5870, 0.1140])
    return np.expand_dims(greyscale, axis=3)

img_height = 32
img_width = 32

image_data = imread('../data/manualTest.png')
image_data = rgb2gray([image_data])
image_shape = image_data.shape
crop_top = (image_shape[1] - img_height) // 2
crop_left = (image_shape[2] - img_width) // 2
plt.imshow(image_data.reshape(image_shape[1], image_shape[2]))
plt.show()
image = image_data[:, crop_top:crop_top+img_height, crop_left:crop_left+img_width, :]

plt.imshow(image.reshape(img_height, img_width))
plt.show()


# image = tf.image.decode_jpeg(tf.read_file('../data/manualTest.png'), channels=1)
# shape = tf.shape(input=image)
# height, width = shape[0], shape[1]
# crop_top = (height - 64) // 2
# crop_left = (width - 64) // 2
# image = tf.slice(image, [crop_top, crop_left, 0], [64, 64, -1])
#
# image = tf.reshape(image, [64, 64, 1])
# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
# image = tf.multiply(tf.subtract(image, 0.5), 2)
# image = tf.image.resize_images(image, [img_height, img_width])
# images = tf.reshape(image, [1, img_height, img_width, 1])


with tf.name_scope("input"):
    # Placeholders for feeding input images
    x = tf.placeholder(tf.float32, shape=(None, img_height, img_width, 1), name='x')

dropout_rates = [1, 1, 1]

logits, y_pred_cls = predict(x, dropout_rates, 1, 11)


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


test_pred = session.run(y_pred_cls, feed_dict={ x: image })
print('Prediction:', test_pred)
