""" Module implementing (ensemble) Kalman filtering. 

In DIESEL, an ensemble is a dask array of shape (n_members, dim).

"""
import dask.array as da
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

    def _update_ensemble(self, ensemble, G, data_std, cov_pushfwd, sqrt):
        """ Helper function for updating the ensemble members over a single period (step).
        This function assumes that the compute intensive intermediate matrices 
        have already been computed.

        Parameters
        ----------
        ensemble: dask.array (n_members, m)
            Ensemble members (one vector per member).
        G: dask.array (n, m)
            Observation operator.
        data_std: float
            Standard deviation of observational noise.
        cov_pushfwd: dask.array (m, n)
            Covariance pushforward cov @ G.T
        sqrt: dask.array (n, n)
            Lower Cholesky factor (square root) of the data covariance.

        Returns
        -------
        update_members: dask.array (n_members, m) (lazy)

        """
        # First compute the inverse of the sqrt.
        inv_sqrt = da.linalg.inv(sqrt)

        # Inverese of the other matrix involved.
        inv_2 = da.linalg.inv(sqrt + data_std * eye(G.shape[0]))

        kalman_gain_tilde = matmul(cov_pushfwd,
                matmul(inv_sqrt.T, inv_2))

        # Compute predictions for each member using batched matrix multiplication.
        base_pred = matmul(G, ensemble[:, :, None]) # Resulting shape (n_members, m, 1)
        ensemble_updated = ensemble[:, :, None] - matmul(kalman_gain_tilde, base_pred)

        # We remove the last dimension before returning.
        return ensemble_updated.squeeze(-1)

    def update_ensemble(self, mean, ensemble, G, y, data_std, cov):
        """ Update an ensemble over a single period (step).

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
        return self._update_mean(mean, G, y, cov_pushfwd, inv), self._update_ensemble(ensemble, G, data_std, cov_pushfwd, sqrt)

    def update_mean_sequential(self, mean, G, y, data_std, cov):
        """ Update the mean over a single period (step) by assimilating the 
        data sequentially (one data point at a time).

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
        mean_updated = mean

        # Loop over the data points and ingest sequentially.
        for i in range(G.shape[0]):
            # One data points.
            G_seq = G[i, :].reshape(1, -1)
            y_seq = y[i].reshape(1, -1)

            mean_updated = self.update_mean(mean, G, y, data_std, cov)   
        return mean_updated

    def update_ensemble_sequential(self, mean, ensemble, G, y, data_std, cov, covariance_estimator=None):
        """ Update an ensemble over a single period (step) by assimilating the 
        data sequentially (one data point at a time).

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
        covariance_estimator: function, defaults to None
            If provided, then at each step the covariance is estimated from 
            the updated ensemble members using the given function. 
            Signature should be ensemble -> covariance matrix.

        Returns
        -------
        update_mean: dask.array (m, 1) (lazy)
        update_members: dask.array (n_members, m) (lazy)

        """
        mean_updated, ensemble_updated = mean, ensemble

        # Loop over the data points and ingest sequentially.
        for i in range(G.shape[0]):
            print(i)
            # One data points.
            G_seq = G[i, :].reshape(1, -1)
            y_seq = y[i].reshape(1, -1)

            # Re-estimate the covariance if estimator provided.
            if covariance_estimator is not None:
                cov_est = covariance_estimator(ensemble_updated)
            else: cov_est = cov

            mean_updated, ensemble_updated = self.update_ensemble(
                    mean_updated, ensemble_updated,
                    G_seq, y_seq,
                    data_std, cov_est)   
            mean_updated, ensemble_updated = mean_updated.compute(), ensemble_updated.compute()
        return da.from_array(mean_updated), da.from_array(ensemble_updated)
