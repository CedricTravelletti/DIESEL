""" Module implementing (ensemble) Kalman filtering. 

In DIESEL, an ensemble is a dask array of shape (n_members, dim).

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
        """ Update the mean over a single period (step).

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

        _, inv = cholesky_invert(to_invert)
        return self._update_mean(mean, G, y, cov_pushfwd, inv)

    def _update_ensemble(self, ensemble, G, cov_pushfwd, sqrt):
        """ Helper function for updating the ensemble members over a single period (step).
        This function assumes that the compute intensive intermediate matrices 
        have already been computed.

        Parameters
        ----------
        ensemble: dask.array (n_members, m)
            Ensemble members (one vector per member).
        G: dask.array (n, m)
            Observation operator.
        cov_pushfwd: dask.array (m, n)
            Covariance pushforward cov @ G.T
        sqrt: dask.array (n, n)
            Lower Cholesky factor (square root) of the data covariance.

        Returns
        -------
        update_members: dask.array (n_members, m) (lazy)

        """
        # First compute the inverse of the sqrt.
        _, inv_sqrt = cholesky_invert(sqrt)

        # Inverese of the other matrix involved.
        _, inv_2 = cholesky_invert(sqrt + data_std * eye(G.shape[0]))

        kalman_gain_tilde = matmul(cov_pushfwd,
                matmul(inv_sqrt.T, inv_2))

        # Compute predictions for each member using batched matrix multiplication.
        base_pred = matmul(G, ensemble[:, :, None]) # Resulting shape (n_members, m, 1)
        ensemble_updated = ensemble[:, :, None] - matmul(kalman_gain_tilde, base_pred)

        # We remove the last dimension before returning.
        return ensemble_updated.squeeze(-1)

    def update_ensemble(self, mean, ensemble, G, y, data_std, cov):
        """ Update the mean over a single period (step).

        Parameters
        ----------
        mean: dask.array (m)
            Vector of mean elements.
        ensemble: dask.array (n_members, m)
            Ensemble members (one vector per member).
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
        update_members: dask.array (n_members, m) (lazy)

        """
        cov_pushfwd = matmul(cov, transpose(G))
        data_cov = data_std**2 * eye(y.shape[0])
        to_invert = matmul(G, cov_pushfwd) + data_cov

        sqrt, inv = cholesky_invert(to_invert)
        return self._update_mean(mean, G, y, cov_pushfwd, inv), self._update_ensemble(ensemble, G, cov_pushfwd, sqrt)
