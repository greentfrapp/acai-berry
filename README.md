# acai-berry

A Tensorflow implementation of Adversarially Constrained Autoencoder Interpolation (ACAI) from:

[Berthelot, David, et al. **"Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer."** arXiv preprint arXiv:1807.07543 (2018).](https://arxiv.org/abs/1807.07543)

*Only for the line toy experiment.*

## Notes

This is NOT the original implementation. The original authors' implementation is [here](https://github.com/brain-research/acai).

~~There seems to be some parts I'm missing since the generated images (see below) appear to be of significantly lower quality than the images shown in the paper (compare to Figure 3 in original paper).~~

Turns out the original implementation used a 2-pixel thick line, as well as -1 for black pixels and 1 for white (the line segment) for the line task. Changing the input to have this distribution gave better-looking images (see [sample_old.png](https://raw.githubusercontent.com/greentfrapp/acai-berry/master/sample_old.png) versus the image below for comparison).

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

![Sample image to reproduce Figure 3 in original paper](https://raw.githubusercontent.com/greentfrapp/acai-berry/master/sample.png)

Refer to `main.py` for more arguments to use with `--train` or `--generate`.

