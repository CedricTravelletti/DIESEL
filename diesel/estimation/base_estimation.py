""" Basic covariance estimation procedures.

"""
from diesel.utils import cov


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
    # Estimate using homemade (float32) implemenation of da.cov
    return cov(ensemble, rowvar=False)

def localize_covariance(base_cov, localization_matrix):
    """ Performs covariance localization.

    """
    return base_cov * localization_matrix
