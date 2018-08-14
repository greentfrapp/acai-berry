"""
Encoder
"""

import tensorflow as tf
import numpy as np

from .BaseModel import BaseModel
from .utils import HeModifiedNormalInitializer


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
				kernel_initializer=HeModifiedNormalInitializer(slope=0.2),
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
				kernel_initializer=HeModifiedNormalInitializer(slope=0.2),
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
			kernel_initializer=HeModifiedNormalInitializer(slope=0.2),
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
			kernel_initializer=HeModifiedNormalInitializer(slope=0.2),
			name='conv_final_2',
		)
		self.outputs = final_conv_2
