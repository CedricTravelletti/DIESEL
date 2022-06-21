""" Dask implementation of the covariance kernels.

"""
import numpy as np
import dask
import dask.array as da
import dask_distance
import dask_distance._utils as utils


# @utils._broadcast_uv_wrapper
def pairwise_euclidean(coords1, coords2):
    return dask_distance.euclidean(coords1, coords2)

class matern32:
    """ Matern 3/2 covariance kernel.

    """
    def __init__(self, lengthscales):
        """ Build Matern 3/2 kernel.

        Parameters
        ----------
        lengthscales: array-like (n_dims)
            Vector of lengthscales for each individual dimension.

        """
        self.lengthscales = lengthscales

    def covariance_matrix(self, coords1, coords2, lengthscales=None):
        """ Compute covariance matrix between two sets of points.

        Parameters
        ----------
        coords1: (m, n_dims) dask.array or Future
            Point coordinates.
        coords2: (n, n_dims) dask.array or Future
            Point coordinates.
        lengthscales_2: array-like (n_dims), defaults to None.
            Can be used to override using the lengthscales of the kernel and use 
            different ones.
    
        Returns
        -------
        covs: (m, n) delayed dask.array
            Pairwise covariance matrix.
    
        """
        if lengthscales is None:
            lengthscales = self.lengthscales

        dists = dask_distance.seuclidean(coords1, coords2, lengthscales**2)
        res = da.multiply(
                1 + np.sqrt(3) * dists,
                da.exp(-np.sqrt(3) * dists))
        return res
