"""Module grouping the methods for Bayesian estimation of covariance matrices."""

import dask.array as da


class InverseWishartPrior:
    """Inverse Wishart prior for covariance matrices."""

    def __init__(self, lazy_scale_matrix, dof):
        self.lazy_scale_matrix = lazy_scale_matrix
        self.dof = dof
        self.dim = lazy_scale_matrix.shape[0]

        if not self.dof > self.dim - 1:
            raise ValueError(
                "The number of degrees of freedom should be strictly greater than p - 1."
            )

    def posterior_mean(self, samples):
        """Compute posterior means given some data.
        The data likelihood is assumed normal, so that we have a conjugate
        pior.

        Parameters
        ----------
        samples: dask.array (n_samples, dims)
            Observed data.

        Returns
        -------
        lazy_posterior_mean: dask.array (dims, dims)
            Posterior covariance matrix (lazy).

        """
        n = samples.shape[0]
        # Note that we use the biased estimate (normalization by 1/N).
        sample_cov = da.cov(samples, rowvar=False, bias=True)

        lazy_posterior_mean = (
            1 / (n + self.dof - self.dim - 1) * (n * sample_cov + self.lazy_scale_matrix)
        )
        return lazy_posterior_mean
