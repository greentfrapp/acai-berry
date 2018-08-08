import tensorflow as tf
import numpy as np

from BaseModel import BaseModel

class ACAI(BaseModel):

	def __init__(self, name='acai', gamma=0.2, lmbda=0.5):
		self.gamma = gamma
		self.lmbda = lmbda
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):

		self.inputs_image_1 = tf.placeholder(
			shape=(None, 32, 32, 1),
			dtype=tf.float32,
			name='inputs_image_1',
		)
		self.inputs_image_2 = tf.placeholder(
			shape=(None, 32, 32, 1),
			dtype=tf.float32,
			name='inputs_image_2',
		)
		self.inputs_alpha = tf.placeholder(
			shape=(None),
			dtype=tf.float32,
			name='inputs_alpha',
		)

		# Reconstruction for x_1
		self.encoder_1 = Encoder(self.inputs_image_1)
		self.decoder_1 = Decoder(self.encoder_1.outputs)
		self.reconstruction_1 = self.decoder_1.outputs

		# Reconstruction for x_2
		self.encoder_2 = Encoder(self.inputs_image_2)
		self.decoder_2 = Decoder(self.encoder_2.outputs)
		self.reconstruction_2 = self.decoder_2.outputs

		# Latent interpolation between x_1 and x_2
		self.latent_interpolation = tf.reshape(self.inputs_alpha, [-1, 1, 1, 1]) * self.encoder_1.outputs + tf.reshape((1 - self.inputs_alpha), [-1, 1, 1, 1]) * self.encoder_2.outputs
		self.decoder_interpolation = Decoder(tf.reshape(self.latent_interpolation, [-1, 2, 2, 16]))
		self.reconstruction_interpolation = self.decoder_interpolation.outputs
		
		# Critic scores
		self.critic_interpolation = Critic(self.reconstruction_interpolation) # This should be alpha
		self.critic_regular_1 = Critic(self.gamma * self.inputs_image_1 + (1 - self.gamma) * self.reconstruction_1) # This should be 0
		self.critic_regular_2 = Critic(self.gamma * self.inputs_image_2 + (1 - self.gamma) * self.reconstruction_2) # This should be 0
		

		# Autoencoder losses
		self.reconstruction_loss_1 = tf.losses.mean_squared_error(labels=self.inputs_image_1, predictions=self.decoder_1.outputs)
		self.reconstruction_loss_2 = tf.losses.mean_squared_error(labels=self.inputs_image_2, predictions=self.decoder_2.outputs)
		regularization_loss = self.lmbda * tf.reduce_mean(self.critic_interpolation.outputs ** 2)
		self.autoencoder_loss = self.reconstruction_loss_1 + self.reconstruction_loss_2 + regularization_loss

		autoencoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/encoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/decoder')
		self.autoencoder_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.autoencoder_loss, var_list=autoencoder_vars)

		# Critic losses
		critic_loss_1 = tf.losses.mean_squared_error(labels=self.inputs_alpha, predictions=self.critic_interpolation.outputs)
		critic_loss_2 = tf.reduce_mean(self.critic_regular_1.outputs ** 2) + tf.reduce_mean(self.critic_regular_2.outputs ** 2)
		self.critic_loss = critic_loss_1 + critic_loss_2

		critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/critic')
		self.critic_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.critic_loss, var_list=critic_vars)
		

class Encoder(BaseModel):

	def __init__(self, inputs, name='encoder'):
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '/'.join(['acai', self.name]))
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):
		running_output = self.inputs
		n_filters = [2, 4, 8, 16]
		for i, filters in enumerate(n_filters):
			conv_1 = tf.layers.conv2d(
				inputs=running_output,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation=None,
				# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name='conv{}_1'.format(i),
			)
			relu_1 = tf.nn.leaky_relu(
				features=conv_1,
				alpha=0.2,
			)
			conv_2 = tf.layers.conv2d(
				inputs=relu_1,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation=None,
				# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name='conv{}_2'.format(i),
			)
			relu_2 = tf.nn.leaky_relu(
				features=conv_2,
				alpha=0.2,
			)
			pool = tf.layers.average_pooling2d(
				inputs=relu_2,
				pool_size=(2, 2),
				strides=(2, 2),
				padding='valid'
			)
			running_output = pool

		final_conv_1 = tf.layers.conv2d(
			inputs=running_output,
			filters=16,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=None,
			# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_final_1',
		)
		final_relu = tf.nn.leaky_relu(
			features=final_conv_1,
			alpha=0.2,
		)
		final_conv_2 = tf.layers.conv2d(
			inputs=final_relu,
			filters=16,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=None,
			# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_final_2',
		)
		self.outputs = final_conv_2

class Decoder(BaseModel):

	def __init__(self, inputs, name='decoder'):
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '/'.join(['acai', self.name]))
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):
		running_output = self.inputs
		n_filters = [16, 8, 4, 2]
		for i, filters in enumerate(n_filters):
			conv_1 = tf.layers.conv2d(
				inputs=running_output,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation=None,
				# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name='conv{}_1'.format(i),
			)
			relu_1 = tf.nn.leaky_relu(
				features=conv_1,
				alpha=0.2,
			)
			conv_2 = tf.layers.conv2d(
				inputs=relu_1,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation=None,
				# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name='conv{}_2'.format(i),
			)
			relu_2 = tf.nn.leaky_relu(
				features=conv_2,
				alpha=0.2,
			)
			upsample = tf.image.resize_images(
				images=relu_2,
				size=(tf.shape(relu_2)[1]*2, tf.shape(relu_2)[2]*2),
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
			)
			running_output = upsample

		final_conv_1 = tf.layers.conv2d(
			inputs=running_output,
			filters=1,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=None,
			# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_final_1',
		)
		final_relu = tf.nn.leaky_relu(
			features=final_conv_1,
			alpha=0.2,
		)
		final_conv_2 = tf.layers.conv2d(
			inputs=final_relu,
			filters=1,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=None,
			# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_final_2',
		)
		self.outputs = final_conv_2
		# self.outputs = tf.sigmoid(final_conv_2)

class Critic(BaseModel):

	def __init__(self, inputs, name='critic'):
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '/'.join(['acai', self.name]))
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):
		running_output = self.inputs
		n_filters = [2, 4, 8, 16]
		for i, filters in enumerate(n_filters):
			conv_1 = tf.layers.conv2d(
				inputs=running_output,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation=None,
				# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name='conv{}_1'.format(i),
			)
			relu_1 = tf.nn.leaky_relu(
				features=conv_1,
				alpha=0.2,
			)
			conv_2 = tf.layers.conv2d(
				inputs=relu_1,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation=None,
				# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name='conv{}_2'.format(i),
			)
			relu_2 = tf.nn.leaky_relu(
				features=conv_2,
				alpha=0.2,
			)
			pool = tf.layers.average_pooling2d(
				inputs=relu_2,
				pool_size=(2, 2),
				strides=(2, 2),
				padding='valid'
			)
			running_output = pool

		final_conv_1 = tf.layers.conv2d(
			inputs=running_output,
			filters=1,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=None,
			# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_final_1',
		)
		final_relu = tf.nn.leaky_relu(
			features=final_conv_1,
			alpha=0.2,
		)
		final_conv_2 = tf.layers.conv2d(
			inputs=final_relu,
			filters=1,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=None,
			# kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			name='conv_final_2',
		)
		running_output = tf.sigmoid(final_conv_2)
		self.outputs = tf.reduce_mean(running_output, axis=[1, 2, 3])
		self.outputs *= 0.5

