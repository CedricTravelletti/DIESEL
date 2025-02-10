"""Non-distributed versions of the Kalman filter.
This is only used for benchmarking agains the distributed version.
"""

import torch
import time
import dask.array as da
import dask.distributed as dd
from diesel.utils.matrix import cross_covariance

torch.set_num_threads(8)
# Select gpu if available and fallback to cpu else.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequentialEnKF:
    def __init__(self):
        pass

    def update_mean_sequential(self, mean, G, y, data_std, cov):
        """Update the mean over a single period (step) by assimilating the
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
        # Import running client.
        from builtins import CLIENT as global_client

        mean_updated = mean

        # Loop over the data points and ingest sequentially.
        for i in range(G.shape[0]):
            # One data points.
            G_seq = G[i, :].reshape(1, -1)
            y_seq = y[i].reshape(1, -1)

            mean_updated = self.update_mean(mean_updated, G_seq, y_seq, data_std, cov)

            # Have to execute once in a while, otherwise graph gets too big.
            if i % 100 == 0:
                mean_updated = global_client.persist(mean_updated)
        return mean_updated

    def update_ensemble_sequential(
        self, mean, ensemble, G, y, data_std, cov, covariance_estimator=None
    ):
        """Update an ensemble over a single period (step) by assimilating the
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
        # Import running client.
        from builtins import CLIENT as global_client

        mean_updated, ensemble_updated = mean, ensemble

        # Loop over the data points and ingest sequentially.
        last_time = time.time()
        for i in range(G.shape[0]):
            # One data points.
            G_seq = G[i, :].reshape(1, -1)
            y_seq = y[i].reshape(1, 1)

            # Re-estimate the covariance if estimator provided.
            if covariance_estimator is not None:
                cov_est = covariance_estimator(ensemble_updated)
            else:
                cov_est = cov

            mean_updated, ensemble_updated = self.update_ensemble(
                mean_updated, ensemble_updated, G_seq, y_seq, data_std, cov_est
            )

            # Have to execute once in a while, otherwise graph gets too big.
            if i % 10 == 0:
                print(i)
                now = time.time()
                elapsed_time = now - last_time
                last_time = now
                print("Time since last persisting: {}.".format(elapsed_time))
                mean_updated = global_client.persist(mean_updated)
                ensemble_updated = global_client.persist(ensemble_updated)
                dd.wait(ensemble_updated)

                # Repatriate locally, so we can cancel running tasks
                # to free the scheduler.
                # TODO: this is not clean and should be solved.
                mean_tmp = mean_updated.compute()
                ensemble_tmp = ensemble_updated.compute()

                # Cancel cached stuff to clean memory.
                global_client.cancel(mean_updated)
                global_client.cancel(ensemble_updated)
                global_client.cancel(cov_est)

                # After cancellation can re-send to the cluster.
                mean_updated = global_client.persist(da.from_array(mean_tmp))
                ensemble_updated = global_client.persist(da.from_array(ensemble_tmp))

        return mean_updated, ensemble_updated


class SequentialEnKFnondask:
    def __init__(self):
        pass

    def _update_anomalies_single_nondask(
        self, mean, ensemble, G, data_std, cov_pushfwd, sqrt
    ):
        """Helper function for updating the ensemble members during non-dask sequential
        updtating. only processes a single data point.

        Parameters
        ----------
        mean: dask.array (m)
            Vector of mean elements.
        ensemble: dask.array (n_members, m)
            Ensemble members (one vector per member).
        G: dask.array (1, m)
            Observation operator.
        data_std: float
            Standard deviation of observational noise.
        cov_pushfwd: dask.array (1, n)
            Covariance pushforward cov @ G.T
        sqrt: float
            Lower Cholesky factor (square root) of the data covariance.

        Returns
        -------
        anomalies_updated: dask.array (n_members, m) (lazy)
            Updated anomalies (deviations from mean). Have to add
            the updated mean to obtain updated ensemble members.

        """
        # Work with anomalies.
        anomalies = ensemble - mean.reshape(-1)[None, :]

        # First compute the inverse of the sqrt.
        inv_sqrt = 1 / sqrt

        inv_2 = 1 / (sqrt + data_std)
        kalman_gain_tilde = (inv_sqrt * inv_2) * cov_pushfwd

        # Compute predictions for each member using batched matrix multiplication.
        base_pred = torch.matmul(
            G, anomalies[:, :, None]
        )  # Resulting shape (n_members, m, 1)
        anomalies_updated = anomalies[:, :, None] - torch.matmul(
            kalman_gain_tilde, base_pred
        )

        # We remove the last dimension before returning.
        return anomalies_updated.squeeze(-1)

    def update_mean_sequential_nondask(self, mean, G, y, data_std, cov):
        """Update the mean over a single period (step) by assimilating the
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
        # Import running client.
        from builtins import CLIENT as global_client

        mean_updated = global_client.compute(mean).result().reshape(-1, 1)
        # Compute pushforward once and for all. extract lines later.
        cov_pushfwd_full = global_client.persist(cov @ da.transpose(G))
        dd.wait(cov_pushfwd_full)

        # Repatriate y to the local process.
        y = global_client.compute(y).result().reshape(-1, 1)
        G = global_client.compute(G).result()

        # Send the important stuff to torch.
        y = torch.from_numpy(y).to(DEVICE).float()
        G = torch.from_numpy(G).to(DEVICE).float()
        mean_updated = torch.from_numpy(mean_updated).to(DEVICE).float()

        # Loop over the data points and ingest sequentially.
        for i in range(G.shape[0]):
            # Every 500 observations, repatriate a chunk of the pushforward
            # and send it to the GPU.
            if i % 500 == 0:
                i_pushfwd_start = i  # The index at which the local pushfwd starts.
                local_pushfwd = global_client.compute(
                    cov_pushfwd_full[:, i : i + 500]
                ).result()
                local_pushfwd = torch.from_numpy(local_pushfwd).to(DEVICE).float()

            # One data points.
            G_seq = G[i, :].reshape(1, -1)
            y_seq = y[i].reshape(1, 1)

            # Now are fully in numpy.
            cov_pushfwd = local_pushfwd[:, i - i_pushfwd_start].reshape(-1, 1)

            data_cov = data_std**2
            to_invert = torch.matmul(G_seq, cov_pushfwd) + data_cov
            inv = 1 / to_invert[0, 0]

            kalman_gain = inv * cov_pushfwd
            prior_misfit = y_seq - torch.matmul(G_seq, mean_updated)
            mean_updated = mean_updated + torch.matmul(kalman_gain, prior_misfit)

        return mean_updated.detach().cpu().numpy()

    def update_ensemble_sequential_nondask(
        self, mean, ensemble, G, y, data_std, localization_matrix
    ):
        """Update the mean over a single period (step) by assimilating the
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
        localization_matrix: dask.array (m, m)
            Matrix used to perform localization. Will get Hadamard-producted with
            the empirical covariance at every assimilation stage.

        Returns
        -------
        update_mean: dask.array (m, 1) (lazy)

        """
        # Import running client.
        from builtins import CLIENT as global_client

        mean_updated = global_client.compute(mean).result().reshape(-1, 1)
        ensemble_updated = global_client.compute(ensemble).result()

        # Repatriate y to the local process.
        y_loc = global_client.compute(y).result().reshape(-1, 1)
        G_loc = global_client.compute(G).result()

        # Send the important stuff to torch.
        y_loc = torch.from_numpy(y_loc).to(DEVICE).float()
        G_loc = torch.from_numpy(G_loc).to(DEVICE).float()
        mean_updated = torch.from_numpy(mean_updated).to(DEVICE).float()
        ensemble_updated = torch.from_numpy(ensemble_updated).to(DEVICE).float()

        # Loop over the data points and ingest sequentially.
        for i in range(G.shape[0]):
            # One data points.
            G_seq = G_loc[i, :].reshape(1, -1)
            y_seq = y_loc[i].reshape(1, 1)

            # Find the indices at which G_seq is non zero and extract those
            # parts of the covariance.
            _, obs_ind = G_seq.nonzero(as_tuple=True)
            obs_ind = obs_ind.cpu().numpy()

            # Extract the concerned line of the empirical covariance.
            cov_pushfwd = cross_covariance(
                ensemble_updated.cpu(), ensemble_updated.cpu()[:, obs_ind], rowvar=False
            ).reshape(-1, 1)

            cov_pushfwd = cov_pushfwd.to(DEVICE).float()
            loc_obs_cov = (
                torch.from_numpy(
                    global_client.compute(localization_matrix[:, obs_ind]).result()
                )
                .to(DEVICE)
                .float()
            )

            cov_pushfwd = torch.mul(cov_pushfwd, loc_obs_cov)

            data_cov = data_std**2
            to_invert = torch.matmul(G_seq, cov_pushfwd) + data_cov
            inv = 1 / to_invert[0]
            sqrt = torch.sqrt(to_invert[0])

            kalman_gain = inv * cov_pushfwd
            prior_misfit = y_seq - torch.matmul(G_seq, mean_updated)

            anomalies_updated = self._update_anomalies_single_nondask(
                mean_updated, ensemble_updated, G_seq, data_std, cov_pushfwd, sqrt
            )
            # Warning, have to update mean after ensemble, since ensemble use the prior mean in the update.
            mean_updated = mean_updated + torch.matmul(kalman_gain, prior_misfit)
            # Add the mean to get ensemble from anomalies.
            ensemble_updated = mean_updated.reshape(-1)[None, :] + anomalies_updated
        return mean_updated.detach().cpu().numpy().reshape(
            -1
        ), ensemble_updated.detach().cpu().numpy()
