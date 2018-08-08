
import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

from models import ACAI
from task import LineTask

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', False, 'Train')

def main(unused_args):
	model = ACAI()
	task = LineTask()
	for i in range(100):
		samples_1, samples_2, alpha = task.next_batch(64)
		feed_dict = {
			model.inputs_image_1: samples_1,
			model.inputs_image_2: samples_2,
			model.inputs_alpha: alpha,
		}
		autoencoder_loss, _ = sess.run([model.autoencoder_loss, autoencoder_optimize], feed_dict)
		critic_loss, _ = sess.run([model.critic_loss, critic_optimize], feed_dict)
		if (i + 1) % 50 == 0:
			print('Step {} - autoencoder_loss: {:.3f} - critic_loss: {:.3f}'.format(i + 1, autoencoder_loss, critic_loss))

if __name__ == '__main__':
	app.run(main)
