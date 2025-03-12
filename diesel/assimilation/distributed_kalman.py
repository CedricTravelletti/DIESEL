"""Module implementing (ensemble) Kalman filtering.

In DIESEL, an ensemble is a dask array of shape (n_members, dim).

"""

# Use torch for the sequential updating (which is done entirely on the scheduler.
import torch
from dask.array import eye, matmul, transpose

from diesel.utils import cholesky_invert, svd_invert

torch.set_num_threads(8)
# Select gpu if available and fallback to cpu else.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsembleKalmanFilter:
    def __init__(self):
        pass

    def _update_mean(self, mean, G, y, cov_pushfwd, inv):
        """Helper function for updating the mean over a single period.
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
        update_mean: dask.array (m) (lazy)

        """
        y = y.reshape(-1, 1)
        mean = mean.reshape(-1, 1)

        kalman_gain = matmul(cov_pushfwd, inv)
        prior_misfit = y - matmul(G, mean)
        mean_updated = mean + matmul(kalman_gain, prior_misfit)
        return mean_updated.reshape(-1)

    def update_mean(self, mean, G, y, data_std, cov):
        """Update the mean over a single period (step).

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
        update_mean: dask.array (m) (lazy)

        """
        cov_pushfwd = matmul(cov, transpose(G))
        data_cov = data_std**2 * eye(y.shape[0])
        to_invert = matmul(G, cov_pushfwd) + data_cov

        _, inv = cholesky_invert(to_invert)
        return self._update_mean(mean, G, y, cov_pushfwd, inv)

    def _update_anomalies(self, mean, ensemble, G, data_std, cov_pushfwd, sqrt, svd_rank=1000):
        """Helper function for updating the ensemble members over a single period (step).
        This function assumes that the compute intensive intermediate matrices
        have already been computed.

        Parameters
        ----------
        mean: dask.array (m)
            Vector of mean elements.
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
        anomalies_updated: dask.array (n_members, m) (lazy)
            Updated anomalies (deviations from mean). Have to add
            the updated mean to obtain updated ensemble members.

        """
        # Import running client.
        from builtins import CLIENT as global_client

        # Work with anomalies.
        anomalies = ensemble - mean.reshape(-1)[None, :]

        # First compute the inverse of the sqrt.
        _, inv_sqrt = svd_invert(sqrt, svd_rank=svd_rank, client=global_client)

        # TODO: Just trying to see where it goes wrong.
        # Inverese of the other matrix involved.
        _, inv_2 = svd_invert(
            sqrt + data_std * eye(G.shape[0]), svd_rank=svd_rank, client=global_client
        )
        kalman_gain_tilde = matmul(cov_pushfwd, matmul(inv_sqrt.T, inv_2))

        # Compute predictions for each member using batched matrix multiplication.
        base_pred = matmul(G, anomalies[:, :, None])  # Resulting shape (n_members, m, 1)
        anomalies_updated = anomalies[:, :, None] - matmul(kalman_gain_tilde, base_pred)

        # We remove the last dimension before returning.
        return anomalies_updated.squeeze(-1)

    def update_ensemble(self, mean, ensemble, G, y, data_std, cov, svd_rank=1000):
        """Update an ensemble over a single period (step).

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
        update_mean: dask.array (m) (lazy)
        update_members: dask.array (n_members, m) (lazy)

        """
        # Import running client.
        from builtins import CLIENT as global_client

        cov_pushfwd = matmul(cov, transpose(G))
        data_cov = data_std**2 * eye(y.shape[0])
        to_invert = matmul(G, cov_pushfwd) + data_cov

        sqrt, inv = svd_invert(to_invert, svd_rank=svd_rank, client=global_client)

        anomalies_updated = self._update_anomalies(
            mean, ensemble, G, data_std, cov_pushfwd, sqrt, svd_rank=svd_rank
        )
        mean_updated = self._update_mean(mean, G, y, cov_pushfwd, inv)

        # Add the mean to get ensemble from anomalies.
        ensemble_updated = mean_updated.reshape(-1)[None, :] + anomalies_updated

        return mean_updated.astype("float32"), ensemble_updated.astype("float32")
