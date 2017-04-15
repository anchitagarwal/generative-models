import os
import time
from datetime import datetime
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
tf.app.flags.DEFINE_string('images_dir', 'images',
										"""Directory where to write generated images.""")
tf.app.flags.DEFINE_string('logdir', 'logdir',
										"""Directory where to write event logs and checkpoint.""")
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
	return tf.reshape(image, [CROP_IMAGE_SIZE, CROP_IMAGE_SIZE, 3])

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
	return tf.subtract(tf.div(
		tf.cast(tf.image.resize_images(
			image_batch, [s_size * 2 ** 4, s_size * 2 ** 4]), 'float'), 127.5), 1.0)
	
def main(argv=None):
	"""
	Main function that calls a training batch sample and trains the model.
	"""
	dcgan = DCGAN(s_size=6)
	traindata = inputs(dcgan.batch_size, dcgan.s_size)
	losses = dcgan.loss(traindata)

	# feature mapping
	graph = tf.get_default_graph()
	features_g = tf.reduce_mean(graph.get_tensor_by_name('dg/d/conv4/outputs:0'), 0)
	features_t = tf.reduce_mean(graph.get_tensor_by_name('dt/d/conv4/outputs:0'), 0)
	# adding the regularization term
	losses[dcgan.g] += tf.multiply(tf.nn.l2_loss(features_t - features_g), 0.05)

	# train and summary
	tf.summary.scalar('g_loss', losses[dcgan.g])
	tf.summary.scalar('d_loss', losses[dcgan.d])
	train_op = dcgan.train(losses, learning_rate=0.001)
	summary_op = tf.summary.merge_all()

	g_saver = tf.train.Saver(dcgan.g.variables)
	d_saver = tf.train.Saver(dcgan.d.variables)
	g_checkpoint_path = os.path.join(FLAGS.logdir, 'g.ckpt')
	d_checkpoint_path = os.path.join(FLAGS.logdir, 'd.ckpt')

	with tf.Session() as sess:
		summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)
		# restore or initialize generator
		sess.run(tf.global_variables_initializer())
		if os.path.exists(g_checkpoint_path):
			print('restore variables:')
			for v in dcgan.g.variables:
				print('  ' + v.name)
			g_saver.restore(sess, g_checkpoint_path)
		if os.path.exists(d_checkpoint_path):
			print('restore variables:')
			for v in dcgan.d.variables:
				print('  ' + v.name)
			d_saver.restore(sess, d_checkpoint_path)

		# setup for monitoring
		sample_z = sess.run(tf.random_uniform([dcgan.batch_size, dcgan.z_dim],
												minval=-1.0,
												maxval=1.0))
		images = dcgan.sample_images(inputs=sample_z)

		# start training
		tf.train.start_queue_runners(sess=sess)

		for step in range(FLAGS.max_steps):
			start_time = time.time()
			_, g_loss, d_loss = sess.run([train_op, losses[dcgan.g], losses[dcgan.d]])
			duration = time.time() - start_time
			print('{}: step {:5d}, loss = (G: {:.8f}, D: {:.8f}) ({:.3f} sec/batch)'.format(
													datetime.now(), step, g_loss, d_loss, duration))

			# save generated images
			if step % 100 == 0:
				# summary
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)
				# sample images
				filename = os.path.join(FLAGS.images_dir, '%05d.jpg' % step)
				with open(filename, 'wb') as f:
					f.write(sess.run(images))

			# save variables
			if step % 500 == 0:
				g_saver.save(sess, g_checkpoint_path, global_step=step)
				d_saver.save(sess, d_checkpoint_path, global_step=step)

if __name__ == '__main__':
	tf.app.run()