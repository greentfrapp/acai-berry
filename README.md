# acai-berry

A Tensorflow implementation of Adversarially Constrained Autoencoder Interpolation (ACAI) from:

[Berthelot, David, et al. **"Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer."** arXiv preprint arXiv:1807.07543 (2018).](https://arxiv.org/abs/1807.07543)

*Only for the line toy experiment.*

## Notes

This is NOT the original implementation. The original authors' implementation is [here](https://github.com/brain-research/acai).

There seems to be some parts I'm missing since the generated images (see below) appear to be of significantly lower quality than the images shown in the paper (see Figure 3 in original paper).

One change I am aware of is using a larger learning rate of 2e-4 for the autoencoder, which I implemented to get better reconstruction.

## Instructions

Just run

```
$ main.py --train
```

Then running

```
$ main.py --generate --start=135 --end=0
```

will generate a row of images (see below) interpolating from the left to the right. The original (non-reconstructed) images are the first and last, while the second and second-last images are the respective reconstructions.

![Sample image to reproduce Figure 3 in original paper](https://raw.githubusercontent.com/greentfrapp/acai-berry/master/sample_1.png)

Another sample with `--start=0` and `--end=-90`:

![Interpolating from 0 to -90 degrees](https://raw.githubusercontent.com/greentfrapp/acai-berry/master/sample_2.png)

The results seem to be rather poor when the difference between start and end is too large (> ~135) eg. `--start=0` and `--end=180`:

![Poor results when attempting to interpolate from 0 to 180 degrees](https://raw.githubusercontent.com/greentfrapp/acai-berry/master/sample_3.png)

Refer to `main.py` for more arguments to use with `--train` or `--generate`.

