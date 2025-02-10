"""Module for sampling from multivariate gaussians."""

import dask.array as da


class SvdSampler:
    """Sample multivariate gaussian from SVD decomposition of
    its covariance matrix.

    Parameters
    ----------
    svd_u: dask_array
        Left singular vectors of the covariance matrix
        (as obtained from da.linalg.svd_compressed).
    svd_s: dask_array
        Vector of singular values of the covariance matrix.

    """

    def __init__(self, svd_u, svd_s):
        self.svd_u, self.svd_s = svd_u, svd_s

        # Equivalent of Cholesky matrix in traditional sampling.
        smat = da.diag(da.sqrt(self.svd_s))
        self.cholesky_lazy = da.dot(self.svd_u, smat)

        self.dim = svd_u.shape[0]

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples: int
            Number of independent samples to generate.

        Returns
        -------
        samples: array [n_samples, dim]

        """
        samples = da.random.normal(size=(n_samples, self.svd_u.shape[1], 1))
        samples = da.dot(self.cholesky_lazy, samples).squeeze().T
        return samples


class CholeskySampler:
    """Non lazy."""

    def __init__(self, covariance_matrix):
        self.cholesky = da.linalg.cholesky(covariance_matrix, lower=True).compute()

    def sample(self, n_samples):
        samples = da.random.normal(size=(n_samples, self.cholesky.shape[0], 1))
        samples = da.dot(self.cholesky, samples).squeeze().T
        return samples
