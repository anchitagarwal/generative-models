import tensorflow as tf

class Generator:
	def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
		"""
		Initializer method for the Generator class.

		Args:
			depths: Python list, number of filters for each conv2d_transpose layers
			s_size: Integer, initial sample size to begin with

		Returns:
			Nothing
		"""
		self.depths = depths + [3]
		self.s_size = s_size
		self.reuse = False

	def __call__(self, inputs, training=False):
		"""
		Calling method for the Generator object. This method builds the generator network.

		Args:
			inputs: Tensor, input latent space vector for the generator
			training: Boolean, whether in training or inference mode

		Returns:
			A Tensor
		"""
		inputs = tf.convert_to_tensor(inputs)
		with tf.variable_scope('g', reuse=self.reuse):
			# reshape input to depths[0] * s_size * s_size
			with tf.variable_scope('reshape'):
				outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
				outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
				outputs = tf.nn.relu(
					tf.layers.batch_normalization(outputs, training=training),
					name='outputs')

			# transposed convolutions x 4
			with tf.variable_scope('deconv1'):
				outputs = tf.layers.conv2d_transpose(
					outputs,
					filters=self.depths[1],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
				outputs = tf.nn.relu(
					tf.layers.batch_normalization(outputs, training=training), name="outputs")

			with tf.variable_scope('deconv2'):
				outputs = tf.layers.conv2d_transpose(
					outputs,
					filters=self.depths[2],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
				outputs = tf.nn.relu(
					tf.layers.batch_normalization(outputs, training=training), name="outputs")

			with tf.variable_scope('deconv3'):
				outputs = tf.layers.conv2d_transpose(
					outputs,
					filters=self.depths[3],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
				outputs = tf.nn.relu(
					tf.layers.batch_normalization(outputs, training=training), name="outputs")

			with tf.variable_scope('deconv4'):
				outputs = tf.layers.conv2d_transpose(
					outputs,
					filters=self.depths[4],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
			
			# output images
			with tf.variable_scope('tanh'):
				outputs = tf.tanh(outputs, name="outputs")

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')

		return outputs

class Discriminator:
	def __init__(self, depths=[64, 128, 256, 512]):
		"""
		Initializer method for the Discriminator class.

		Args:
			depths: Python list, number of filters for each conv2d layers

		Returns:
			Nothing
		"""
		self.depths = [3] + depths
		self.reuse = False

	def __call__(self, inputs, training=False, name=''):
		"""
		Calling method for the Discriminator object. This method builds the discriminator network.

		Args:
			inputs: Tensor, input latent space vector for the discriminator
			training: Boolean, whether in training or inference mode
			name: String, placeholder for generated or real images

		Returns:
			A Tensor
		"""
		def leaky_relu(x, leak=0.2, name=''):
			"""
			Method that implements leaky ReLU.

			Args:
				x: Tensor, input before activation
				leak: Float, leak parameter
				name: String, placeholder for generated or real images

			Returns:
				A Tensor
			"""
			return tf.maximum(x, x * leak, name=name)

		outputs = tf.convert_to_tensor(inputs)

		with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
			# colvolution x 4
			with tf.variable_scope('conv1'):
				outputs = tf.layers.conv2d(
					outputs,
					filters=self.depths[1],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
				outputs = leaky_relu(
					tf.layers.batch_normalization(outputs, training=training), name="outputs")

			with tf.variable_scope('conv2'):
				outputs = tf.layers.conv2d(
					outputs,
					filters=self.depths[2],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
				outputs = leaky_relu(
					tf.layers.batch_normalization(outputs, training=training), name="outputs")

			with tf.variable_scope('conv3'):
				outputs = tf.layers.conv2d(
					outputs,
					filters=self.depths[3],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
				outputs = leaky_relu(
					tf.layers.batch_normalization(outputs, training=training),name="outputs")

			with tf.variable_scope('conv4'):
				outputs = tf.layers.conv2d(
					outputs,
					filters=self.depths[4],
					kernel_size=[5,5],
					strides=[2,2],
					padding="same")
				outputs = leaky_relu(
					tf.layers.batch_normalization(outputs, training=training), name="outputs")

			with tf.variable_scope('classify'):
				batch_size = outputs.get_shape()[0].value
				reshape = tf.reshape(outputs, [batch_size, -1])
				outputs = tf.layers.dense(reshape, 2, name='outputs')

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')

		return outputs

class DCGAN:
	def __init__(self,
				 batch_size=128, z_dim=100, s_size=4,
				 g_depths=[1024, 512, 256, 128],
				 d_depths=[64, 128, 256, 512]):
		"""
		Initialization method for DCGAN class.

		Args:
			batch_size: Integer, batch size
			z_dim: Integer, dimension of the latent space
			s_size: Integer, dimension of the first volume of deconv layer
			g_depths: Python list, list containing number of filters for each layer in generator
			d_depths: Python list, list containing number of filters for each layer in discriminator

		Returns:
			Nothing
		"""
		self.batch_size = batch_size
		self.z_dim = z_dim
		self.s_size = s_size
		self.g = Generator(depths=g_depths, s_size=self.s_size)
		self.d = Discriminator(depths=d_depths)
		self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

	def loss(self, traindata):
		"""
		Method that builds the model and calculates losses.

		Args:
			inputdata: 4-D Tensor of shape `[batch_size, height, width, channels]`.

		Returns:
			Dictionary of each model losses.
		"""
		generated = self.g(self.z, training=True)
		g_outputs = self.d(generated, training=True, name='g')
		t_outputs = self.d(traindata, training=True, name='t')

		# add each losses to collection
		tf.add_to_collection(
			'g_losses',
			tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=tf.ones([self.batch_size], dtype=tf.int64),logits=g_outputs)))

		tf.add_to_collection(
			'd_losses',
			tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=tf.zeros([self.batch_size], dtype=tf.int64),logits=g_outputs)))

		tf.add_to_collection(
			'd_losses',
			tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=tf.ones([self.batch_size], dtype=tf.int64),logits=t_outputs)))

		return {
			self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_losses'),
			self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_losses')
		}

	def train(self, losses, learning_rate=0.0002, beta1=0.5):
		"""
		Method that trains the model for the training dataset.

		Args:
			losses: Python dictionary, generator and discriminator losses.
			learning_rate: Float, learning rate for the model.
			beta1: Float, exponential decay rate for the 1st moment estimates.

		Returns:
			train op
		"""
		g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
		d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
		g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
		d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)
		with tf.control_dependencies([g_opt_op, d_opt_op]):
			return tf.no_op(name="train")

	def sample_images(self, row=8, col=8, inputs=None):
		"""
		Method that draws sample images from generator and displays them.

		Args:
			row: Integer
			col: Integer
			inputs: Tensor, input latent feature vector

		Returns:
			8x8 grid of generated images
		"""
		if inputs is None:
			inputs = self.z
		images = self.g(inputs, training=True)
		images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
		images = [image for image in tf.split(images, self.batch_size, axis=0)]
		rows = []
		for i in range(row):
			rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
		image = tf.concat(rows, 1)
		return tf.image.encode_jpeg(tf.squeeze(image, [0]))