""" Module implementing Kalman filtering.

"""
from dask.array import matmul, eye, transpose
from diesel.utils import cholesky_invert


class EnsembleKalmanFilter:
    def __init__(self):
        pass

    def _update_mean(self, mean, G, y, cov_pushfwd, inv):
        """ Helper function for updating the mean over a single period.
        This function assumes that the compute intensive intermediate matrices 
        have already been computed.

        Parameters
        ----------
        mean: dask.array (m)
            Vector of mean elements.
        G: dask.array (n, m)
            Observation operator.
        y: dask.array (n, 1)
            Observed data.
        cov_pushfwd: dask.array (m, n)
            Covariance pushforward cov @ G.T
        inv: dask.array (n, n)
            Inverse intermediate matrix.

        Returns
        -------
        update_mean: dask.array (m, 1) (lazy)

        """
        y = y.reshape(-1, 1)
        mean = mean.reshape(-1, 1)

        kalman_gain = matmul(cov_pushfwd, inv)
        prior_misfit = y - matmul(G, mean)
        mean_updated = mean + matmul(kalman_gain, prior_misfit)
        return mean_updated

    def update_mean(self, mean, G, y, data_std, cov):
        """ Helper function for updating the mean over a single period.
        This function assumes that the compute intensive intermediate matrices 
        have already been computed.

        Parameters
        ----------
        mean: dask.array (m)
            Vector of mean elements.
        G: dask.array (n, m)
            Observation operator.
        y: dask.array (n, 1)
            Observed data.
        data_std: float
            Standard deviation of observational noise.
        cov: dask.array (m, m)
            Covariance matrix (estimated) between the grid points. 
            Can be lazy.

        Returns
        -------
        update_mean: dask.array (m, 1) (lazy)

        """
        cov_pushfwd = matmul(cov, transpose(G))
        data_cov = data_std**2 * eye(y.shape[0])
        to_invert = matmul(G, cov_pushfwd) + data_cov

        cholesky, inv = cholesky_invert(to_invert)
        return self._update_mean(mean, G, y, cov_pushfwd, inv)
