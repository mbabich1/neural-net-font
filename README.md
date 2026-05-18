# Neural Net Font

## Introduction

This repository is a basic exploration into generating rasterized text
based on open source font families. We are using Noto Sans and Noto
Serif in regular, bold, italics, and bold-italics. This was made for
one of the CMSC 678 group projects at UMBC in the Spring of 2026.

## Design

We rasterize text with the `freetype-py` bindings to the FreeType
library directly onto appropriately-sized Pytorch tensors. This
approach is suitable for English words in the Latin alphabet, but
other languages in other scripts may require an even higher level
library.

We visualize the text with `matplotlib` and use `kagglehub`, `pandas`,
`numpy`, and `torch` to acquire and work with the data.

For the neural network, we use convoluted kernels on the data and then
apply a simple generative adversarial network (GAN) design, which is
based on having a generator network and a discriminator network.
Memory and runtime constraints are the main limiting factors with our
approach. Comparing different implementations of GANs, comparing the
GAN's results to other approaches (such as diffusion), or running the
GAN on much larger compute resources are all potential points of
follow up. Additionally, with more compute resources, other metadata
could be provided other than just the font, such as encoded tokens of
the word or perhaps training on tokens instead of on words.

Initially, we considered using sentences, but words were sufficient
for the scope of the project to show results in our limited memory and
compute budget. For words, we used a [370k English word corpus from
Kaggle](https://www.kaggle.com/datasets/ruchi798/part-of-speech-tagging/data)
and we kept it in the entirely lower case form of the dataset.

A future exploration of similar concepts should, at a minimum, look
into turning "word" into "Word" and "WORD" capitalized and uppercase
variations, but as it would triple the size of the dataset, we chose
to use 8 different fonts rather than to use permutations on one word
in the same font. We also chose to keep everything at size 12 and
rendered starting on the upper left corner of the image tensor.

## Installation instructions

To install this code so it can run, follow these instructions while
inside the top level directory (where this README is located):

1. Run `python -m venv .venv` to create a Python venv.
2. Run `source .venv/bin/activate` to enter the venv.
3. Run `pip install --editable .` to locally install all of the
   dependencies as specified in the `pyproject.toml`. Nvidia CUDA is
   strongly recommended to be used with Pytorch.
