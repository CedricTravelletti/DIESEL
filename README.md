<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/CedricTravelletti/DIESEL/blob/main/logo.png?raw=true" width="60%" alt="DIESEL" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  Developed using <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Deepseek" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>

</div>
---

# DIESEL
DIESEL is a package for DIstributed EStimation of EnsembLe covariance matrices.

It leverages distributed arrays (DASK) to allow handling very large matrices and implements distributed versions of covariance estimation algorithms.

See the [project website](https://cedrictravelletti.github.io/DIESEL/) for more information.

## Installation
This project uses the very fast [uv](https://docs.astral.sh/uv/) virtual environment and package manager. A virtual environment 
containing all needed dependencies can be created using:
```
uv sync
```
One can then activate the environment using:
```
source .venv/bin/activate
```

## Available Functionalities


- **Gridding**:
  - `SquareGrid`: Build and manipulate square grids.

- **Covariance Estimation**:
  - `matern32`: Construct Matern 3/2 covariance matrices.
  - `empirical_covariance`: Estimate covariance using empirical covariance of ensembles.

- **Bayesian Estimation**:
  - Various Bayesian estimation methods.

- **Ensemble Kalman Filtering**:
  - Implement ensemble Kalman filtering algorithms.

- **Sampling**:
  - `SvdSampler`: Sample using SVD of the covariance matrix.
  - `CholeskySampler`: Sample using Cholesky decomposition.

- **Scoring**:
  - Compute various scoring metrics for probabilistic forecasts.

- **Plotting**:
  - Tools for visualizing grids and ensembles.

## Missing Functionalities

- **Extended Documentation**:
  - More detailed API documentation.
  - Tutorials and example notebooks.

- **Additional Testing**:
  - Comprehensive unit tests for all modules.
  - Integration tests.

- **Advanced Covariance Models**:
  - Implement additional covariance models.

- **Performance Optimization**:
  - Further optimization for large-scale computations.

- **User Guides**:
  - Detailed user guides and best practices.

## Roadmap 
- [ ] demonstration of covariance estimation on ERA5 (ensemble members created artificially)
- [ ] high-resolution covariance regularization 
- [ ] high-resolution data assimilation 
- [ ] scaling tests (workers) 

