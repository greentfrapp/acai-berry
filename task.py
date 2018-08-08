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

	def next_batch(self, batchsize):
		angles = np.random.rand(batchsize) * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]
		images = np.zeros((batchsize, self.uncropped_height, self.uncropped_width))
		images[:, int(self.uncropped_width / 2):int(self.uncropped_width / 2 + self.length), int(self.uncropped_width / 2)] = 1
		image = images[0]
		crop_dim = (
			(self.uncropped_width - self.width) / 2,
			(self.uncropped_height - self.height) / 2,
			(self.uncropped_width - self.width) / 2 + self.width,
			(self.uncropped_height - self.height) / 2 + self.height,
		)
		samples_1 = []
		for angle in angles:
			samples_1.append(np.array(Image.fromarray(image).rotate(angle).crop(crop_dim)))
		samples_2 = []
		for angle in angles:
			angle += 180
			samples_2.append(np.array(Image.fromarray(image).rotate(angle).crop(crop_dim)))
		alpha = np.random.rand(batchsize) * 0.5
		return np.array(samples_1).reshape(-1, self.height, self.width, 1), np.array(samples_2).reshape(-1, self.height, self.width, 1), alpha
