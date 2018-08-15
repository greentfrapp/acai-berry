import tensorflow as tf
import numpy as np

from .BaseModel import BaseModel
from .Encoder import Encoder
from .Decoder import Decoder
from .Critic import Critic


class ACAI(BaseModel):

	def __init__(self, name='acai'):
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

	def build_model(self):

		self.inputs_image = tf.placeholder(
			shape=(None, 32, 32, 1),
			dtype=tf.float32,
			name='inputs_image',
		)
		self.inputs_alpha = tf.placeholder(
			shape=(None),
			dtype=tf.float32,
			name='inputs_alpha',
		)

		self.gamma = tf.placeholder(
			shape=(),
			dtype=tf.float32,
			name='gamma',
		)
		self.lmbda = tf.placeholder(
			shape=(),
			dtype=tf.float32,
			name='lmbda',
		)

		# Reconstruction for x_1
		self.encoder = Encoder(self.inputs_image)
		self.decoder = Decoder(self.encoder.outputs)
		self.reconstruction = self.decoder.outputs

		# Latent interpolation between x_1 and x_2
		self.latent_interpolation = tf.reshape(self.inputs_alpha, [-1, 1, 1, 1]) * self.encoder.outputs + tf.reshape((1 - self.inputs_alpha), [-1, 1, 1, 1]) * self.encoder.outputs[::-1]
		self.decoder_interpolation = Decoder(tf.reshape(self.latent_interpolation, [-1, 2, 2, 16]))
		self.reconstruction_interpolation = self.decoder_interpolation.outputs
		
		# Critic scores
		self.critic_interpolation = Critic(self.reconstruction_interpolation) # This should be alpha
		self.critic_regularization = Critic(self.gamma * self.inputs_image + (1 - self.gamma) * self.reconstruction) # This should be 0

		# Autoencoder losses
		self.reconstruction_loss = tf.losses.mean_squared_error(labels=self.inputs_image, predictions=self.decoder.outputs)
		autoencoder_regularization_loss = self.lmbda * tf.reduce_mean(self.critic_interpolation.outputs ** 2)
		self.autoencoder_loss = self.reconstruction_loss + autoencoder_regularization_loss

		autoencoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/encoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/decoder')
		self.autoencoder_optimize = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.autoencoder_loss, var_list=autoencoder_vars)

		# Critic losses
		critic_error_loss = tf.losses.mean_squared_error(labels=self.inputs_alpha, predictions=self.critic_interpolation.outputs)
		critic_regularization_loss = tf.reduce_mean(self.critic_regularization.outputs ** 2)
		self.critic_loss = critic_error_loss + critic_regularization_loss

		critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'acai/critic')
		self.critic_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.critic_loss, var_list=critic_vars)
