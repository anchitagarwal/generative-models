import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from dcgan import DCGAN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 5000,
										"""number of examples for train""")
tf.app.flags.DEFINE_integer('max_steps', 10001,
										"""Number of batches to run.""")
tf.app.flags.DEFINE_string('data_dir', '/home/anchit/lsun/',
										"""Path to LSUN dataset.""")
tf.app.flags.DEFINE_string('file_pattern', '/home/anchit/lsun/*.webp',
										""""File pattern for the images.""")

CROP_IMAGE_SIZE = 96

def read_lsun(filename_queue):
	"""
	Reads and parses examples from LSUN dataset.

	Args:
		filename_queue: A queue of strings with filenames to read from

	Returns:
		An object representing a single example
	"""
	# read an entire image file
	image_reader = tf.WholeFileReader()
	# read a file from the queue
	_, image_file = image_reader.read(filename_queue)
	# decode the image as a JPEG
	image = tf.image.decode_jpeg(image_file)
	image = tf.image.resize_image_with_crop_or_pad(image, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE)
	return image

def inputs(batch_size, s_size):
	"""
	Construct input for LSUN dataset using Reader op.

	Args:
		batch_size: Integer, required batch size
		s_size: Integer, desired crop image factor

	Returns:
		image: 4-D Tensor of shape ``[batch_size, height, width, channels]``
	"""
	# make a queue of file names including all the images files in the directory
	filename_queue = tf.train.string_input_producer(
								tf.train.match_filenames_once(FLAGS.file_pattern))
	image = read_lsun(filename_queue)
	min_queue_examples = FLAGS.num_examples_per_epoch_for_train
	capacity = min_queue_examples + 3 * batch_size
	image_batch = tf.train.shuffle_batch(
										 [image],
										 batch_size=batch_size,
										 capacity=capacity,
										 min_after_dequeue=min_queue_examples)
	tf.summary.image('images', image_batch)
	return tf.subtract(
		tf.div(tf.image.resize_images(image_batch, [s_size * 2 ** 4, s_size * 2 ** 4]), 127.5), 1.0)
	
def main(argv=None):
	"""
	Main function that calls a training batch sample and trains the model.
	"""
	dcgan = DCGAN(s_size=6)
	traindata = inputs(dcgan.batch_size, dcgan.s_size)
	losses = dcgan.loss(traindata)