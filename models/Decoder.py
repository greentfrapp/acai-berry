"""
Decoder
"""

import tensorflow as tf
import numpy as np

from .BaseModel import BaseModel
from .utils import HeModifiedNormalInitializer


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
			kernel_initializer=HeModifiedNormalInitializer(slope=0.2),
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
			kernel_initializer=HeModifiedNormalInitializer(slope=0.2),
			name='conv_final_2',
		)
		self.outputs = final_conv_2
