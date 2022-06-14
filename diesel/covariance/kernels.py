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

def matern32(coords, lambda0):
    """ Matern 3/2 covariance kernel.

    Parameters
    ----------
    coords: (n_pts, n_dims) dask.array or Future
        Point coordinates.

    Returns
    -------
    covs: (n_pts, n_pts) delayed dask.array
        Pairwise covariance matrix.

    """
    dists = pairwise_euclidean(coords, coords)
    res = da.multiply(
            1 + (np.sqrt(3) / lambda0) * dists,
            da.exp(-(np.sqrt(3) / lambda0) * dists))
    return res
