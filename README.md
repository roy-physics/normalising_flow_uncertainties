# normalising_flow_uncertainties

This repository contains code and related materials for the paper **"Systematic Uncertainties and Data Complexity in Normalizing Flows"**. In this work, we investigate the interplay between data variance and initialization variance in normalizing flow analyses using astrophysical datasets.

## Overview

Normalizing flows (NFs) are powerful tools for inferring probability distributions from finite samples, relevant across the physical sciences. Using two toy astrophysical examples (a Plummer sphere potential and a Miyamoto-Nagai disk potential), we explore the relative contributions of two sources of uncertainty:

1. **Data Variance**: Varying the draws from the training distribution.
2. **Initialization Variance**: Varying the neural network's initial conditions.

Our results show that:
- For simple distributions, initialization variance dominates.
- For more complex distributions (such as perturbed datasets with substructure), data variance takes precedence as measured by the Kullback-Leibler divergence.

## Code Files

- **`generate_MN_disk_particles.py`**: 
  This script initializes particles and evolves them in a Miyamoto-Nagai disk potential using `galpy`. Before running the `init_variance` and `data_variance` files for disk and stream particles, this needs to be run and the output particle distribution saved. 

- **`init_variance_disk_with_stream.py`**: 
  This script initializes the Miyamoto-Nagai disk model with a stellar stream perturbation. It handles the setup and sampling from the disk and stream distribution, which is used to study the variance in normalizing flows when training on complex astrophysical distributions.

- **`data_variance_disk_with_stream.py`**: 
  This script generates training and validation data for the Miyamoto-Nagai disk potential and perturbs the dataset with a toy model of a stellar stream. This is used to quantify the data variance when training normalizing flows on this perturbed dataset.

- **`init_variance_plummer_with_stream.py`** and **`data_variance_plummer_with_stream.py`**:
  These scripts are identical to their disk counterparts but they use Plummer sphere data as the base distributions. The Plummer sphere data needed for training is generated within these scripts. 

- **`kl_divergence_and_visualisations.ipynb`**:
  This notebook visualises the Plummer sphere and disk data distributions and calculates the KL-divergence between the base distributions and the perturbed distributions (i.e. adding a stream).

## Installation

To run the code, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- nflows (for normalizing flows)
- numpy
- galpy (for galactic dynamics)

You can install the dependencies via pip:
```bash
pip install torch nflows numpy galpy
