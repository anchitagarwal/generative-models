import os
import sys
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from dcgan import DCGAN
import load_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 50001, """Number of batches to run.""")
tf.app.flags.DEFINE_string('data_dir', '/home/anchit/lsun/', """Path to LSUN dataset.""")
tf.app.flags.DEFINE_string('images_dir', 'images', """Directory where to write generated images.""")
tf.app.flags.DEFINE_string('logdir', 'logdir', 
	"""Directory where to write event logs and checkpoint.""")
	
def main(argv=None):
	"""
	Main function that calls a training batch sample and trains the model.
	"""
	if len(argv) < 2:
		print "Please input desired dataset in cmd line: `lsun` or `celeb`."
		sys.exit()

	dcgan = DCGAN(batch_size=64, s_size=4)

	traindata = None
	if argv[1] == 'lsun':
		# load input pipeline for LSUN dataset
		traindata = load_data.lsun_inputs(dcgan.batch_size, dcgan.s_size)
	elif argv[1] == 'celeb':
		# load input pipeline for CelebA dataset
		traindata = load_data.celeb_inputs(dcgan.batch_size, dcgan.s_size)

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
	train_op = dcgan.train(losses, learning_rate=0.0002)
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