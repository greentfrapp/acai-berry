import tensorflow as tf
import numpy as np

from BaseModel import BaseModel

class ACAI(BaseModel):

	def __init__(self, name='acai'):
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

		gamma = 0.2
		lmbda = 0.5

		self.encoder_1 = Encoder(self.inputs_image_1)
		self.encoder_2 = Encoder(self.inputs_image_2)
		self.latent_interpolation = self.inputs_alpha * self.encoder_1.outputs + (1 - self.inputs_alpha) * self.encoder_2.outputs
		self.decoder_1 = Decoder(self.encoder_1.outputs)
		self.decoder_2 = Decoder(self.encoder_2.outputs)
		self.decoder_interpolation = Decoder(self.interpolation)
		self.critic_interpolation = Critic(self.decoder_interpolation.outputs)
		self.critic_regular = Critic(gamma * self.inputs_image_1 + (1 - gamma) * self.decoder_1.outputs)

		critic_loss_1 = tf.losses.mean_squared_error(labels=self.inputs_alpha, predictions=self.critic_interpolation.outputs)
		critic_loss_2 = self.critic_regular ** 2
		critic_loss = critic_loss_1 + critic_loss_2
		
		reconstruction_loss = tf.losses.mean_squared_error(labels=self.inputs_image, predictions=self.decoder.outputs)
		regularization_loss = lmda * self.critic_interpolation ** 2
		autoencoder_loss = reconstruction_loss + regularization_loss

		critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/critic')
		critic_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(critic_loss, var_list=critic_vars)
		autoencoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/encoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/decoder')
		autoencoder_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(critic_loss, var_list=autoencoder_vars)

class Encoder(BaseModel):

	def __init__(self, inputs, name='encoder'):
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
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
				name='conv{}_2'.format(i),
			)
			# No Leaky ReLU for last layer
			if i == len(n_filters) - 1:
				relu_2 = conv_2
			else:
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
		self.outputs = running_output

class Decoder(BaseModel):

	def __init__(self, inputs, name='decoder'):
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):
		running_output = self.inputs
		n_filters = [8, 4, 2, 1]
		for i, filters in enumerate(n_filters):
			conv_1 = tf.layers.conv2d(
				inputs=running_output,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding='same',
				activation=None,
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
				name='conv{}_2'.format(i),
			)
			# No Leaky ReLU for last layer
			relu_2 = tf.nn.leaky_relu(
				features=conv_2,
				alpha=0.2,
			)
			upsample = tf.image.resize_images(
				images=relu_2,
				size=(tf.shape(relu_2)[1]*2, tf.shape(relu_2)[2]*2),
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
			)
			running_output = pool

		final_conv_1 = tf.layers.conv2d(
			inputs=running_output,
			filters=1,
			kernel_size=(3, 3),
			strides=(1, 1),
			padding='same',
			activation=None,
			name='conv{}_2'.format(i),
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
			name='conv{}_2'.format(i),
		)

		self.outputs = final_conv_2

class Critic(BaseModel):

	def __init__(self, inputs, name='encoder'):
		self.inputs = inputs
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
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
				name='conv{}_2'.format(i),
			)
			# No Leaky ReLU for last layer
			if i == len(n_filters) - 1:
				relu_2 = conv_2
			else:
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
		self.outputs = tf.reduce_mean(running_output, axis=[1, 2, 3])

