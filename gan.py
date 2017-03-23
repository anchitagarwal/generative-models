import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import os

class network(object):
	def __init__(self):
		self.mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
		self.build_model()
		self.train()

	def xavier_init(self, size):
		"""
		Method that initializes weights and biases in Xavier format.

		Args:
			size: shape of the weight vector.

		Returns:
			weight vector of random initializations in Xavier format.
		"""
		input_dim = size[0]
		std_dev = 1 / tf.sqrt(input_dim / 2.)
		return tf.random_normal(shape=size, mean=0.0, stddev=std_dev)

	def generator(self, z):
		"""
		Generator method that takes noise as input and learns the pdf of data.

		Args:
			z: Latent noise.

		Returns:
			G_prob: pdf(x)
		"""
		G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
		G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
		G_prob = tf.nn.sigmoid(G_log_prob)

		return G_prob

	def discriminator(self, x):
		"""
		Discriminator method that takes fake/real images and outputs the probablities.

		Args:
			x: input image.

		Returns:
			D_prob: probability that image is real or fake.
			D_logit
		"""
		D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
		D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
		D_prob = tf.nn.sigmoid(D_logit)

		return D_prob, D_logit

	def sample_noise(self, size):
		"""
		Method that generates noisy samples of shape ``size``.

		Args:
			size: [m, n] => m = # of samples; n = dimensions of feature vectors.

		Returns:
			Random numbers of shape ``size`` generated from U(-1, 1)
		"""
		return np.random.uniform(-1., 1., size=size)

	def plot(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

		return fig

	def build_model(self):
		"""
		Method that builds Generator's and Discriminator's weights and biases.
		"""
		# discriminator net
		self.X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

		self.D_W1 = tf.Variable(self.xavier_init([784, 128]), name='D_W1')
		self.D_b1 = tf.Variable(tf.zeros([128]), name='D_b1')
		self.D_W2 = tf.Variable(self.xavier_init([128, 1]), name='D_W2')
		self.D_b2 = tf.Variable(tf.zeros([1]), name='D_b2')

		self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

		# generative model
		self.Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

		self.G_W1 = tf.Variable(self.xavier_init([100, 128]), name='G_W1')
		self.G_b1 = tf.Variable(tf.zeros([128]), name='G_b1')
		self.G_W2 = tf.Variable(self.xavier_init([128, 784]), name='G_W2')
		self.G_b2 = tf.Variable(tf.zeros([784]), name='G_b2')

		self.theta_G = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

	def train(self):
		"""
		Method to trains GAN.
		"""
		# get real/fake prob and logit
		G_sample = self.generator(self.Z)
		D_real, D_logit_real = self.discriminator(self.X)
		D_fake, D_logit_fake = self.discriminator(G_sample)

		# calculate generator and discriminator loss
		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
										logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
										logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
		D_loss = D_loss_real + D_loss_fake
		G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
										logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

		# discriminator and generator optimizers
		D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=self.theta_D)
		G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=self.theta_G)

		mini_batch_size = 128
		Z_dim = 100

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		if not os.path.exists('out/'):
			os.makedirs('out/')

		i = 0

		for it in range(1000000):
			if it % 1000 == 0:
				samples = sess.run(G_sample, feed_dict={self.Z: self.sample_noise([16, Z_dim])})

				fig = self.plot(samples)
				plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
				i += 1
				plt.close(fig)

			X_mb, _ = self.mnist.train.next_batch(mini_batch_size)

			_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict=
								{self.X: X_mb, self.Z: self.sample_noise([mini_batch_size, Z_dim])})
			_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict=
								{self.Z: self.sample_noise([mini_batch_size, Z_dim])})

			if it % 1000 == 0:
				print('Iter: {}'.format(it))
				print('D loss: {:.4}'.format(D_loss_curr))
				print('G_loss: {:.4}'.format(G_loss_curr))
				print()

if __name__ == "__main__":
	gan_network = network()