import numpy as np
from PIL import Image


class LineTask():

	def __init__(self):
		self.angle_range = [0, 360]
		self.length = 16
		self.height = 32
		self.width = 32
		self.uncropped_height = 50
		self.uncropped_width = 50

	def next_batch(self, batchsize, start=None, end=None):
		if start is not None and end is not None and batchsize == 2:
			angles = [start%360, end%360]
		else:
			angles = np.random.rand(batchsize) * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]
		image = np.zeros((self.uncropped_height, self.uncropped_width))
		image[int(self.uncropped_height / 2) - 1:int(self.uncropped_height / 2) + 1, int(self.uncropped_width / 2):int(self.uncropped_width / 2 + self.length),] = 1
		crop_dim = (
			(self.uncropped_width - self.width) / 2,
			(self.uncropped_height - self.height) / 2,
			(self.uncropped_width - self.width) / 2 + self.width,
			(self.uncropped_height - self.height) / 2 + self.height,
		)
		samples = []
		for angle in angles:
			samples.append(np.array(Image.fromarray(image).rotate(angle).crop(crop_dim)))
		alpha = np.random.rand(batchsize) * 0.5
		return np.array(samples).reshape(-1, self.height, self.width, 1) * 2 - 1, alpha
