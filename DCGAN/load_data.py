import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 5000, 
	"""number of examples for train""")
tf.app.flags.DEFINE_string('file_pattern_lsun', '/home/anchit/datasets/lsun/*.webp',
	"""File pattern for the LSUN images.""")
tf.app.flags.DEFINE_string('file_pattern_celeb', 
	'/home/anchit/datasets/CelebA/Img/img_align_celeba_png/*.png',
	"""File pattern for the CelebA images""")

CROP_IMAGE_SIZE = 64

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
	return tf.reshape(image, [CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 3])

def lsun_inputs(batch_size, s_size):
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
		tf.train.match_filenames_once(FLAGS.file_pattern_lsun))
	image = read_lsun(filename_queue)
	min_queue_examples = FLAGS.num_examples_per_epoch_for_train
	capacity = min_queue_examples + 3 * batch_size
	image_batch = tf.train.shuffle_batch(
		[image],
		batch_size=batch_size,
		capacity=capacity,
		min_after_dequeue=min_queue_examples)
	tf.summary.image('images', image_batch)
	return tf.subtract(tf.div(tf.cast(tf.image.resize_images(
		image_batch, [s_size * 2 ** 4, s_size * 2 ** 4]), 'float'), 127.5), 1.0)

def read_celeb(filename_queue):
	"""
	Reads and parses examples from CelebA dataset.

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
	image = tf.image.decode_png(image_file)
	image = tf.image.resize_image_with_crop_or_pad(image, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE)
	return tf.reshape(image, [CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 3])

def celeb_inputs(batch_size, s_size):
	"""
	Construct input for CelebA dataset using Reader op.

	Args:
		batch_size: Integer, required batch size
		s_size: Integer, desired crop image factor

	Returns:
		image: 4-D Tensor of shape ``[batch_size, height, width, channels]``
	"""
	# make a queue of file names including all the images files in the directory
	filename_queue = tf.train.string_input_producer(
		tf.train.match_filenames_once(FLAGS.file_pattern_celeb))
	image = read_celeb(filename_queue)
	min_queue_examples = FLAGS.num_examples_per_epoch_for_train
	capacity = min_queue_examples + 3 * batch_size
	image_batch = tf.train.shuffle_batch(
		[image],
		batch_size=batch_size,
		capacity=capacity,
		min_after_dequeue=min_queue_examples)
	tf.summary.image('images', image_batch)
	return tf.subtract(tf.div(tf.cast(tf.image.resize_images(
		image_batch, [s_size * 2 ** 4, s_size * 2 ** 4]), 'float'), 127.5), 1.0)