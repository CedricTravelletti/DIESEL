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

def haversine(point1, point2):
    """ Haversine distance between two points.

    Parameters
    ----------
    point1: array (2,)
        Point coordinates, with the first element being the latitude in degrees
        and the second one the longitude.
    point2: array (2,)

    Returns
    -------
    dist: float
        Distance in kilometers.

    """
    lat1, lon1 = np.deg2rad(point1[0]), np.deg2rad(point1[1])
    lat2, lon2 = np.deg2rad(point2[0]), np.deg2rad(point2[1])

    angle = np.arccos(
            np.sin(lat1) * np.sin(lat2)
            + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    km_conversion = 6371
    return km_conversion * angle

def pairwise_haversine(coords1, coords2):
    return dask_distance.cdist(coords1, coords2, haversine)

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

    def covariance_matrix(self, coords1, coords2, lengthscales=None, metric='euclidean'):
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
            Note that for haversine metric one should provide only one lengthscale.
        metric: 'euclidean' or 'haversine'.
    
        Returns
        -------
        covs: (m, n) delayed dask.array
            Pairwise covariance matrix.
    
        """
        if lengthscales is None:
            lengthscales = self.lengthscales

        if metric == 'euclidean':
            dists = dask_distance.seuclidean(coords1, coords2, lengthscales**2)
        elif metric == 'haversine':
            dists = (1 / lengthscales) * pairwise_haversine(coords1, coords2)
        else: raise ValueError("Metric not implemented.")

        res = da.multiply(
                1 + np.sqrt(3) * dists,
                da.exp(-np.sqrt(3) * dists))
        return res

class squared_exponential:
    """ Squared exponential covariance kernel.

    """
    def __init__(self, lengthscales):
        """ Build squared_exponential kernel.

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
        res = da.exp(- (1 / 2) * dists**2)
        return res
