# JAX Tutorial: Bayesian Neural Networks (BNN) with Ensemble Kalman Inversion (EKI) for Bayesian Inference
This Jupyter Notebook demonstrates inference with Ensemble Kalman Inversion for Bayesian Neural Networks (BNNs) using JAX, a high-performance machine learning library, for efficient and flexible numerical computation. We will focus on this tutorial on regression, however, this method can be extended to other ML tasks (see [Here](https://arxiv.org/abs/1808.03620))

# Table of Contents
[Prerequisites](#prerequisites) 

[Installation](#installation)

[Introduction](#introduction)  

# Prerequisites
To follow this tutorial, you should have a basic understanding of:

- Probability
- Python programming
- Linear algebra
- Neural networks
- JAX

For a general guide on how to utilize JAX, see [here](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).

# Installation
To install the required packages, run the following command:
`pip install -r requirements.txt`

Additionally, for instructions on how to install the GPU version of JAX, see the installation guide [here](https://github.com/google/jax#installation).


# Introduction
Bayesian Neural Networks (BNNs) provide a principled approach to quantify uncertainties in deep learning models. BNNs perform Bayesian inference on the weights of the neural networks, resulting in a probability distribution over the weights instead of point estimates. In this tutorial, we showcase how to implement BNNs using the JAX library and perform inference using the Ensemble Kalman Inversion (EKI) method.

![BNN figure](https://drive.google.com/uc?id=1ZFCZsQ_uNmHsyVnplZYbbURsS9qiolg5)


Ensemble Kalman Inversion (EKI) is a popular class of methods that utilize the Ensemble Kalman Filter (EnKF) to estimate a set of unknown model parameters from model outputs. These methods are gradient-free, easily parallelizable, and scale well in high-dimension inverse problems with ensemble sizes much smaller than the total number of parameters. For BNNs, we assume our unknown parameters $\boldsymbol{\xi}\in\mathbb{R}^{N_\xi}$ are the BNN weights and these parameters have a prior distribution $p(\boldsymbol{\xi})\sim\mathcal{N}(0,1)$. Additionally, the observations $\mathbf{y}\in\mathbb{R}^{N_y}$, correspond to the outputs of interest, and are related to the BNN weights through the evaluation of the neural network operator $\mathcal{G}$. We define and ensemble of $J$ neural networks and represent the BNN as a series of NN samples $\{\boldsymbol{\xi}^{(j)}_0\}$ initially, which are updated via the EnKF update equations seen in the algorithm below. Here $\mathcal{G}_i$ represents the evaluation of the neural network on the minibatch at iteration $i$.

![EKI Pseudocode](https://drive.google.com/uc?id=1Oe4bEptEmjFEF5L1OHUPiNyLs7CJzcVc)


The implementation for this approach is loosely based off the following references:

- [Ensemble Kalman Inversion: A Derivative-Free Technique For Machine Learning Tasks
](https://arxiv.org/abs/1808.03620)
- [The Ensemble Kalman Filter for Inverse Problems](https://arxiv.org/abs/1209.2736)
- [Iterated Kalman Methodology For Inverse Problems](https://arxiv.org/abs/2102.01580)
- [Efficient Bayesian Physics Informed Neural Networks for Inverse Problems via Ensemble Kalman Inversion](https://arxiv.org/pdf/2303.07392.pdf)
