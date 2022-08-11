# Surrogate Assisted Multiobjective Optimisation

This repository consists of the work done on the [ISCMI 2021 paper](https://ieeexplore.ieee.org/document/9654934) extension submitted to Neural Computing and Applications (NCAA).

## Installation

The repository is forked and extended from the base work by [roprisor](https://github.com/roprisor/cvxportfolio) and the [CVXPortfolio Group](https://github.com/cvxgrp/cvxportfolio).
### Custom Cvxportfolio Environment

All results and calculations in the submitted manuscript were obtain using the following custom build virtual environment:

```bash
#Dependencies are version specific
conda create -n cvxport python=3.8
conda activate cvxport
conda install pytables
conda install seaborn
conda install multiprocess
conda install jupyter
conda install scikit-learn
conda install pandas=0.25
conda install numpy=1.19
conda install -c anaconda cython
pip install deap
pip install setuptools==58.0

#The modified code for the experiments is found here
pip install git+https://github.com/tvanzyl/cvxportfolio.git
pip install git+https://github.com/tvanzyl/pymoo.git
pip install git+https://github.com/tvanzyl/pysamoo.git
```

Optional and not required:

```bash
#For a pymoo speedup:
sudo apt install build-essential
git clone https://github.com/tvanzyl/pymoo
cd pymoo
make compile
pip install .
```