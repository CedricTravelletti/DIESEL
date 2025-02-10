"""Dask implementation of the covariance kernels."""

import numpy as np
import dask.array as da
from diesel.dask_distance import euclidean, cdist, seuclidean
from haversine import haversine


# @_broadcast_uv_wrapper
def pairwise_euclidean(coords1, coords2):
    return euclidean(coords1, coords2)


def pairwise_haversine(coords1, coords2):
    return cdist(coords1, coords2, lambda x, y: haversine(x[0], x[1], y[0], y[1]))


class matern32:
    """Matern 3/2 covariance kernel."""

    def __init__(self, lengthscales):
        """Build Matern 3/2 kernel.

        Parameters
        ----------
        lengthscales: array-like (n_dims)
            Vector of lengthscales for each individual dimension.

        """
        self.lengthscales = lengthscales

    def covariance_matrix(
        self, coords1, coords2=None, lengthscales=None, metric="euclidean"
    ):
        """Compute covariance matrix between two sets of points.

        Parameters
        ----------
        coords1: (m, n_dims) dask.array or Future
            Point coordinates.
        coords2: (n, n_dims) dask.array or Future
            Point coordinates.
        lengthscales: array-like (n_dims), defaults to None.
            Can be used to override using the lengthscales of the kernel and use
            different ones.
            Note that for haversine metric one should provide only one lengthscale.
        metric: 'euclidean' or 'haversine'.

        Returns
        -------
        covs: (m, n) delayed dask.array
            Pairwise covariance matrix.

        """
        if coords2 is None:
            coords2 = coords1

        if lengthscales is None:
            lengthscales = self.lengthscales

        if metric == "euclidean":
            dists = seuclidean(coords1, coords2, lengthscales**2)
        elif metric == "haversine":
            dists = (1 / lengthscales) * pairwise_haversine(coords1, coords2)
        else:
            raise ValueError("Metric not implemented.")

        res = da.multiply(
            1 + np.sqrt(3, dtype=np.float32) * dists,
            da.exp(-np.sqrt(3, dtype=np.float32) * dists),
            dtype="float32",
        )
        return res


class squared_exponential:
    """Squared exponential covariance kernel."""

    def __init__(self, lengthscales):
        """Build squared_exponential kernel.

        Parameters
        ----------
        lengthscales: array-like (n_dims)
            Vector of lengthscales for each individual dimension.

        """
        self.lengthscales = lengthscales

    def covariance_matrix(
        self, coords1, coords2, lengthscales=None, metric="euclidean"
    ):
        """Compute covariance matrix between two sets of points.

        Parameters
        ----------
        coords1: (m, n_dims) dask.array or Future
            Point coordinates.
        coords2: (n, n_dims) dask.array or Future
            Point coordinates.
        lengthscales: array-like (n_dims), defaults to None.
            Can be used to override using the lengthscales of the kernel and use
            different ones.

        Returns
        -------
        covs: (m, n) delayed dask.array
            Pairwise covariance matrix.

        """
        if lengthscales is None:
            lengthscales = self.lengthscales

        if metric == "euclidean":
            dists = seuclidean(coords1, coords2, lengthscales**2)
        elif metric == "haversine":
            dists = (1 / lengthscales) * pairwise_haversine(coords1, coords2)
        else:
            raise ValueError("Metric not implemented.")

        res = da.exp(-(1 / 2) * dists**2)
        return res
