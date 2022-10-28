# Unifying Approaches in Data Subset Selection - Experiments

See https://arxiv.org/abs/2208.00549.

> The mutual information between predictions and model parameters -- also referred to as expected information gain or BALD in machine learning -- measures informativeness. It is a popular acquisition function in Bayesian active learning and Bayesian optimal experiment design. In data subset selection, i.e. active learning and active sampling, several recent works use Fisher information, Hessians, similarity matrices based on the gradients, or simply the gradient lengths to compute the acquisition scores that guide sample selection. Are these different approaches connected, and if so how? In this paper, we revisit the Fisher information and use it to show how several otherwise disparate methods are connected as approximations of information-theoretic quantities.

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

