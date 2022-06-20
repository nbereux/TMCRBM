# Tethered Monte-Carlo Sampling for Restricted Boltzmann Machines

This repository contains the material used to obtain the results in our [paper](https://arxiv.org/abs/2206.01310v1) 

## Prerequisites

Install python 3:

```bash
sudo apt update
sudo apt install python3
sudo apt install python3-pip
```

Install [torch 1.11.0](https://pytorch.org/get-started/locally/) 

Install requirements:
```bash
pip install -r requirements.txt
```

## Installation 

Install the package:
```bash
pip install -e .
```

## Usage

You can import the package and use the functions in the `rbm` module. For example:
```python
from rbm.models import RBM
```

An example of how to train a RBM using the tethered monte-carlo sampling method is shown in those notebooks :
- [Dataset 1D 2 clusters](./notebook/train_1d2c.ipynb)
- [Dataset 2D 3 clusters](./notebook/train_2d3c.ipynb)
  
And a sampling and analysis example is shown in those notebooks :
- [Dataset 1D 2 clusters](./notebook/sample_1d2c.ipynb)
- [Dataset 2D 3 clusters](./notebook/sample_2d3c.ipynb)

