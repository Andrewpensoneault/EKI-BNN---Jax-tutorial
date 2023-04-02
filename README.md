# JAX Tutorial: Bayesian Neural Networks (BNN) with Ensemble Kalman Inversion (EKI) for Bayesian Inference
This Jupyter Notebook demonstrates inference with Ensemble Kalman Inversion for Bayesian Neural Networks (BNNs) using JAX, a high-performance machine learning library, for efficient and flexible numerical computation. 

# Table of Contents
[Introduction](#introduction)  
[Prerequisites](#prerequisites)  
[Installation](#installation)

# Introduction
Bayesian Neural Networks (BNNs) provide a principled approach to quantify uncertainties in deep learning models. BNNs perform Bayesian inference on the weights of the neural networks, resulting in a probability distribution over the weights instead of point estimates. In this tutorial, we showcase how to implement BNNs using the JAX library and perform inference using the Ensemble Kalman Inversion (EKI) method.

The Ensemble Kalman Inversion (EKI) is a computationally efficient algorithm for performing approximate Bayesian Inference. It is based on the Ensemble Kalman Filter (EnKF) and has been proven to be effective for high-dimensional inverse problems. EKI approximates the posterior distribution of the weights using an ensemble of samples and iteratively updates these samples to better match the observed data.

![BNN figure](https://drive.google.com/uc?id=1ZFCZsQ_uNmHsyVnplZYbbURsS9qiolg5)

The implementation for this approach is loosely based off the following references:

- [Ensemble Kalman Inversion: A Derivative-Free Technique For Machine Learning Tasks
](https://arxiv.org/abs/1808.03620)
- [The Ensemble Kalman Filter for Inverse Problems](https://arxiv.org/abs/1209.2736)
- [Iterated Kalman Methodology For Inverse Problems](https://arxiv.org/abs/2102.01580)
- [Efficient Bayesian Physics Informed Neural Networks for Inverse Problems via Ensemble Kalman Inversion](https://arxiv.org/pdf/2303.07392.pdf)


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

