
import tensorflow as tf
import numpy as np
from absl import flags
from absl import app
from PIL import Image

from models import ACAI
from tasks import LineTask

FLAGS = flags.FLAGS

flags.DEFINE_bool('train', False, 'Train')
flags.DEFINE_bool('generate', False, 'Generate')

# --train parameters
flags.DEFINE_integer('steps', 10000, 'Number of training steps')
flags.DEFINE_string('savepath', 'saved_models/', 'Path to save or load models')
flags.DEFINE_integer('batchsize', 64, 'Training batchsize')

# --generate parameters
flags.DEFINE_integer('intervals', 10, 'Number of intervals to interpolate (includes start and end')
flags.DEFINE_integer('start', None, 'Starting angle in degrees, where 0 refers to straight right, counting clockwise')
flags.DEFINE_integer('end', None, 'Ending angle in degrees, where 0 refers to straight right, counting clockwise')


def main(unused_args):

	if FLAGS.train:
		model = ACAI()
		task = LineTask()
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		gamma = 0.2
		lmbda = 0.5
		for step in range(1, FLAGS.steps + 1):
			samples_1, samples_2, alpha = task.next_batch(max(FLAGS.batchsize, 1))
			feed_dict = {
				model.inputs_image_1: samples_1,
				model.inputs_image_2: samples_1[::-1],
				model.inputs_alpha: alpha,
				model.gamma: gamma,
				model.lmbda: lmbda,
			}
			autoencoder_loss, recon_loss, _, recon = sess.run([model.autoencoder_loss, model.reconstruction_loss, model.autoencoder_optimize, model.decoder_1.outputs], feed_dict)
			critic_loss, critic_score, _ = sess.run([model.critic_loss, model.critic_interpolation.outputs, model.critic_optimize], feed_dict)
			if step % 100 == 0:
				print('Step #{}'.format(step))
				print('Autoencoder Loss: {:.4f} - Reconstruction Loss: {:.4f}'.format(autoencoder_loss, recon_loss))
				print('Critic Loss: {:.4f} - Average Critic Score: {:.3f}'.format(critic_loss, np.mean(critic_score)))
				# print((np.max(recon), np.min(recon)))
				model.save(sess, FLAGS.savepath, step)

	elif FLAGS.generate:
		model = ACAI()
		task = LineTask()
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		model.load(sess, FLAGS.savepath, verbose=True)
		samples_1, samples_2, _ = task.next_batch(1, start=FLAGS.start, end=FLAGS.end)
		intervals = max(2, FLAGS.intervals)
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
		# Image.fromarray(samples * 255).resize((samples.shape[1] * scale, samples.shape[0] * scale)).convert('RGBA').save('sample.png')

if __name__ == '__main__':
	app.run(main)
