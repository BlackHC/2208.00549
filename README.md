# Unifying Approaches in Data Subset Selection - Experiments

See https://openreview.net/forum?id=UVDAKQANOW and https://arxiv.org/abs/2208.00549.

> Recently proposed methods in data subset selection, that is active learning and active
sampling, use Fisher information, Hessians, similarity matrices based on gradients, and
gradient lengths to estimate how informative data is for a model’s training. Are these
different approaches connected, and if so, how? We revisit the fundamentals of Bayesian
optimal experiment design and show that these recently proposed methods can be understood
as approximations to information-theoretic quantities: among them, the mutual information
between predictions and model parameters, known as expected information gain or BALD in
machine learning, and the mutual information between predictions of acquisition candidates
and test samples, known as expected predictive information gain. We develop a comprehensive
set of approximations using Fisher information and observed information and derive a unified
framework that connects seemingly disparate literature. Although Bayesian methods are often
seen as separate from non-Bayesian ones, the sometimes fuzzy notion of “informativeness”
expressed in various non-Bayesian objectives leads to the same couple of information
quantities, which were, in principle, already known by Lindley (1956) and MacKay (1992).

## Setup

We provide an `enviorment.yml` file for setting up the conda environment. The code was tested with Python 3.9 and PyTorch 1.12.1.

## Overview

The repository is a subset of a more complete active learning repository we will strive to release at a later date.

We use the amazing Laplace package for the Laplace approximation. See https://aleximmer.github.io/Laplace/. 

## Citation

Kirsch, Andreas, and Yarin Gal. "Unifying Approaches in Data Subset Selection via Fisher Information and Information-Theoretic Quantities." arXiv preprint arXiv:2208.00549 (2022).

BibTex:
```bibtex
@misc{kirsch2022unifying,
    title={Unifying Approaches in Data Subset Selection via Fisher Information and Information-Theoretic Quantities},
    author={Andreas Kirsch and Yarin Gal},
    year={2022},
    eprint={2208.00549},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

