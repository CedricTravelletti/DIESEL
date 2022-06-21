""" Implementation of non-stationary Gaussian process models.

"""
import dask.array as da


class BaCompositeGP:
    """ Composite non-stationary GP model as defined in Ba and Joseph (2012).

    """
    def __init__(self, global_covariance, local_covariance):
        self.global_covariance = global_covariance
        self.local_covariance = local_covariance
        self.n_iter_vs = 5

    def _compute_helper_matrices(self, pred_pts, dat_pts, y, vs_data, lmbda):
        """ Compute the matrices involced in the global and local prediction. 
        This function centralizes computations that are shared across the different 
        prediction scenarios.

        pred_pts: dask.array (m, n_dims)
            Coordinates of the prediction points.
        dat_pts: dask.array (n, n_dims)
            Coordinates of the data points.
        y: dask.array (n ,1)
            Observed data.
        vs_data: dask.array (n, 1)
            Local variance scalings at the data points.

        Returns
        -------
        G_cov_mat
        L_cov_mat
        G_cross_cov
        L_cross_cov
        Sigma_sqrt
        inv

        """
        y = y.reshape(-1, 1)

        G_cov_mat = self.global_covariance.covariance_matrix(dat_pts, dat_pts)
        L_cov_mat = self.local_covariance.covariance_matrix(dat_pts, dat_pts)

        G_cross_cov = self.global_covariance.covariance_matrix(pred_pts, dat_pts)
        L_cross_cov = self.local_covariance.covariance_matrix(pred_pts, dat_pts)

        Sigma_sqrt = da.diag(da.sqrt(vs_data.reshape(-1)))

        inv = da.linalg.inv(G_cov_mat + lmbda * Sigma_sqrt @ L_cov_mat @ Sigma_sqrt)
        return G_cov_mat, L_cov_mat, G_cross_cov, L_cross_cov, Sigma_sqrt, inv

    def predict(self, pred_pts, dat_pts, y, lmbda, b):
        """ Compute prediction given some data. 
        The local variances are estimated iteratively in an inner loop.

        This function returns the global and local part of the prediction separately. 
        The complete prediction is the sum of both.

        Parameters
        ----------
        pred_pts: dask.array (m, n_dims)
            Coordinates of the prediction points.
        dat_pts: dask.array (n, n_dims)
            Coordinates of the data points.
        y: dask.array (n ,1)
            Observed data.
        lmbda: float
            Ratio of the local variance to the global variance.
        b: float
            Tuning parameter for the local variances.

        Returns
        -------
        pred_global: dask.array (m, 1)
            Prediction at the given prediction points (global part).
        pred_local: dask.array (m, 1)
            Prediction at the given prediction points (local part).

        """
        y = y.reshape(-1, 1)

        # Initial guess for the vs is ones.
        vs_data = da.ones(y.shape)

        # Iteratively estimate the local variances vs.
        for i in range(self.n_iter_vs):
            # Estimate the global predictor at the data points.
            pred_global_data = self.predict_global(dat_pts, dat_pts, y, vs_data, lmbda)
            vs_pred, vs_data = self.estimate_vs(pred_pts, dat_pts, y, pred_global_data, b)
            print(vs_data.compute())

        # Compute final predictions and return
        # Get matrices needed for prediction.
        (G_cov_mat, L_cov_mat, G_cross_cov,
                L_cross_cov, Sigma_sqrt, inv) = self._compute_helper_matrices(
                        pred_pts, dat_pts, y, vs_data, lmbda)
        # Estimate global mean.
        ones = da.ones(y.shape)
        mu_hat = (
                da.linalg.inv(ones.T @ inv @ ones)
                @
                ones.T @ inv @ y)

        # Estimate the predictors.
        misfit = y - mu_hat * ones 
        pred_global = mu_hat + G_cross_cov @ inv @ misfit
        pred_local = lmbda * da.sqrt(vs_pred) * L_cross_cov @ Sigma_sqrt @ inv @ misfit

        # Trigger computations.
        pred_global = pred_global.compute()
        pred_local = pred_local.compute()

        return pred_global, pred_local

    def predict_global(self, pred_pts, dat_pts, y, vs_data, lmbda):
        """ Fit the global part of the composite GP, for a fixed vector 
        of local variance scalings vs.

        Parameters
        ----------
        pred_pts: dask.array (m, n_dims)
            Coordinates of the prediction points.
        dat_pts: dask.array (n, n_dims)
            Coordinates of the data points.
        y: dask.array (n ,1)
            Observed data.
        vs_data: dask.array (m, 1)
            Local variance scalings at the data points.
        lmbda: float
            Ratio of the local variance to the global variance.

        Returns
        -------
        pred_global: dask.array (m, 1)
            Prediction at the given prediction points (global part).

        """
        y = y.reshape(-1, 1)
        # Get matrices needed for prediction.
        (G_cov_mat, L_cov_mat, G_cross_cov,
                L_cross_cov, Sigma_sqrt, inv) = self._compute_helper_matrices(
                        pred_pts, dat_pts, y, vs_data, lmbda)

        # Estimate global mean.
        ones = da.ones(y.shape)
        mu_hat = (
                da.linalg.inv(ones.T @ inv @ ones)
                @
                ones.T @ inv @ y)

        # Estimate the global predictor.
        misfit = y - mu_hat * ones 
        pred_global = mu_hat + G_cross_cov @ inv @ misfit
        return pred_global

    def estimate_vs(self, pred_pts, dat_pts, y, pred_global_data, b):
        """ Estimate the v(x) local variance scaling using eq (18) 
        from Ba and Joseph (2012). Returns the local variances 
        at the data points and at the prediction points.

        Parameters
        ----------
        pred_pts: dask.array (m, n_dims)
            Coordinates of the prediction points.
        dat_pts: dask.array (n, n_dims)
            Coordinates of the data points.
        y: dask.array (n ,1)
            Observed data.
        pred_global_data: dask.array (n, 1)
            Global prediction at the data points.
        b: float

        Returns
        -------
        vs_pred: dask.array (m, 1)
            Estimated local variance scalings at the prediction points.
        vs_data: dask.array (n, 1)
            Estimated local variance scalings at the data points.

        """
        s_2 = (y - pred_global_data)**2

        # One needs the original (global) covariance model, but 
        # with lengthscales scaled by b.
        mod_lengthscales = b * self.global_covariance.lengthscales

        gb_pred = self.global_covariance.covariance_matrix(
                pred_pts, dat_pts,
                lengthscales=mod_lengthscales)
        gb_data = self.global_covariance.covariance_matrix(
                dat_pts, dat_pts,
                lengthscales=mod_lengthscales)

        ones = da.ones((dat_pts.shape[0], 1))

        vs_pred = gb_pred @ s_2 / gb_pred @ ones
        vs_data = gb_data @ s_2 / gb_data.T @ ones

        # Normalize
        vs_pred = vs_pred / vs_data.mean()
        vs_data = vs_data / vs_data.mean()

        return vs_pred, vs_data
