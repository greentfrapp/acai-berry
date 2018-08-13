# acai-berry

A Tensorflow implementation of ACAI from:

[Berthelot, David, et al. **"Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer."** arXiv preprint arXiv:1807.07543 (2018).](https://arxiv.org/abs/1807.07543)

*Only for the line toy experiment.

Note that this is NOT the original implementation. The original authors' implementation is [here](https://github.com/brain-research/acai).

## Instructions

Just run

```
$ main.py --train
```

Then running

```
$ main.py --generate
```

will generate a row of images interpolating from the left to the right. The original (non-reconstructed) images are the first and last, while the second and second-last images are the respective reconstructions.

