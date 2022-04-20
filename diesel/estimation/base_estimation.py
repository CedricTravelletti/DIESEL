""" Basic covariance estimation procedures.

"""
import dask.array as da


def empirical_covariance(ensemble):
    """ Compute the emprirical covariance of an ensemble.

    Parameters
    ----------
    ensemble: dask.array [n_members, dim]
        Independent realizations of a dim-dimensional random field.

    Returns
    -------
    covariance: dask.array (lazy) [dim, dim]

    """
    return da.cov(ensemble, rowvar=False)
