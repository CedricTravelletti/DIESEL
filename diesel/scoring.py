""" Scoring functions to evaluate quality of probabilistic forecasts.

"""
import dask.array as da


def compute_RE_score(mean_prior, mean_updated, reference):
    """ Reduction of error skill score.
    This score compares a base prediction (mean_prior) with an enhanced prediction (mean_updated). 
    If the enhanced prediction predicts the reference better than the base one, then the score 
    is > 0, the score being 1 if the reconstruction is perfect.

    See Valler et al., Impact of different estimations of the background-error covariance matrix on climate reconstructions based on data assimilation (2019).

    Parameters
    ----------
    mean_prior: dask.array (m)
        Vector of mean elements prior to the updating.
    mean_updated: dask.array (m)
        Vector of mean elements after updating.
    reference: dask.array (m)
        Ground truth to be reconstructed.

    Returns
    -------
    RE_score: dask.array (m)
        Vector of RE scores at each location.

    """
    mean_prior, mean_updated, reference = mean_prior.reshape(-1, 1), mean_updated.reshape(-1, 1), reference.reshape(-1, 1)
    return 1 - (mean_updated - reference)**2 / (mean_prior - reference)**2

def compute_CRPS(ensemble, reference):
    """ Computes the continuous ranked probability score (CRPS).
    This scores evaluates how well a probabilistic forecast (given by an ensemble) 
    predicts a given reference. The CRPS is relative in the sense that it is used to 
    compare different forecasts, lower score being better.

    See Jordan et al., Evaluating Probabilistic Forecasts with scoringRule (2018).

    Parameters
    ----------
    ensemble: dask.array (n_members, m)
        Collection of prediction vectors.
    reference: dask.array (m)
        Ground truth to be reconstructed.

    Returns
    -------
    CRPS: dask.array (m)
        Vector of CRPS scores at each location.

    """
    n_members = ensemble.shape[0]
    accuracy = da.fabs(ensemble - reference.reshape(-1)[None, :]).sum(axis=0)
    spread = da.fabs(ensemble[None, :, :] - ensemble[:, None, :]).sum(axis=0).sum(axis=0)
    CRPS = (1 / n_members) * accuracy - (1 / (2 * n_members**2)) * spread
    return CRPS, accuracy, spread
