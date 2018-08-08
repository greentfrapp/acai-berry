
import tensorflow as tf
import numpy as np
from absl import flags
from absl import app
from PIL import Image

from models import ACAI
from task import LineTask

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', False, 'Train')
flags.DEFINE_bool('generate', False, 'Generate')

flags.DEFINE_integer('steps', 5000, 'Number of training steps')
flags.DEFINE_string('savepath', 'saved_models/', 'Path to save or load models')

def main(unused_args):

	if FLAGS.train:
		model = ACAI()
		task = LineTask()
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		min_autoencoder_loss = np.inf
		for step in range(1, FLAGS.steps + 1):
			samples_1, samples_2, alpha = task.next_batch(64)
			feed_dict = {
				model.inputs_image_1: samples_1,
				model.inputs_image_2: samples_2,
				model.inputs_alpha: alpha,
			}
			autoencoder_loss, recon_loss_1, recon_loss_2, _, recon = sess.run([model.autoencoder_loss, model.reconstruction_loss_1, model.reconstruction_loss_2, model.autoencoder_optimize, model.decoder_1.outputs], feed_dict)
			critic_loss, _ = sess.run([model.critic_loss, model.critic_optimize], feed_dict)
			critic_score = sess.run(model.critic_interpolation.outputs, feed_dict)
			if step % 50 == 0:
				print('Step {} - autoencoder loss: {:.3f} - critic loss: {:.3f} - average critic score: {:.3f}'.format(step, autoencoder_loss, critic_loss, np.mean(critic_score)))
				print((np.max(recon), np.min(recon)))
				print(recon_loss_1)
				print(recon_loss_2)
				model.save(sess, FLAGS.savepath, step)
	elif FLAGS.generate:
		model = ACAI()
		task = LineTask()
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		model.load(sess, FLAGS.savepath, verbose=True)
		samples_1, samples_2, _ = task.next_batch(1)
		intervals = 10
		interpolations = []

		for i in range(intervals + 1):
			alpha = 1 - i / intervals
			feed_dict = {
				model.inputs_image_1: samples_1,
				model.inputs_image_2: samples_2,
				model.inputs_alpha: [alpha],
			}
			interpolation = sess.run(model.decoder_interpolation.outputs, feed_dict).reshape(32, 32)
			interpolations.append(interpolation)
		samples = np.concatenate([samples_1[0].reshape(32, 32)] + interpolations + [samples_2[0].reshape(32, 32)], axis=1)
		scale = 2
		Image.fromarray(samples * 255).resize((samples.shape[1] * scale, samples.shape[0] * scale)).show()


if __name__ == '__main__':
	app.run(main)
