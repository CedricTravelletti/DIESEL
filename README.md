# DIESEL
DIESEL is a package for DIstributed EStimation of EnsembLe covariance matrices.

It leverages distributed arrays (DASK) to allow handling very large matrices and implements distributed versions of covariance estimation algorithms.

See the [project website](https://cedrictravelletti.github.io/DIESEL/) for more information.

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

## Installation

To install DIESEL, use pip:

```bash
pip install diesel
```
