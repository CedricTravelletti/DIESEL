""" Scoring functions to evaluate quality of probabilistic forecasts.

"""
import numpy as np
import dask.array as da


def compute_RE_score(mean_prior, mean_updated, reference, min_lat=-90, max_lat=90):
    """ Reduction of error skill score.
    This score compares a base prediction (mean_prior) with an enhanced prediction (mean_updated). 
    If the enhanced prediction predicts the reference better than the base one, then the score 
    is > 0, the score being 1 if the reconstruction is perfect.

    Note that this score averages over times steps and produces a spatial map.

    See Valler et al., Impact of different estimations of the background-error covariance matrix on climate reconstructions based on data assimilation (2019).

    Parameters
    ----------
    mean_prior: dask.array (m, t)
        Vector of mean elements prior to the updating.
    mean_updated: dask.array (m,t)
        Vector of mean elements after updating.
    reference: xarray.Dataset (m, t)
        Ground truth to be reconstructed.
        Should be provided in dataset format in order to include 
        spatial information.
    min_lat: float, defaults to None.
        If specified, ignore the refions of low latitude 
        in the computation of the mismatch.
    max_lat: float, defaults to None.
        If specified, ignore the refions of high latitude 
        in the computation of the mismatch.

    Returns
    -------
    RE_score: dask.array (m)
        Vector of RE scores at each location.

    """
    # Filter out high/low latitude if provided.
    lat_filter_inds = (reference.latitude < max_lat).data & (reference.latitude > min_lat).data
    reference = reference.data

    mean_prior = mean_prior[lat_filter_inds]
    mean_updated = mean_updated[lat_filter_inds]
    reference = reference[lat_filter_inds]

    # Get rid of Nans.
    mean_prior = mean_prior[~np.isnan(reference)]
    mean_updated = mean_updated[~np.isnan(reference)]
    reference = reference[~np.isnan(reference)]

    # Make sure shapes agree.

    mean_prior, mean_updated, reference = mean_prior.reshape(-1), mean_updated.reshape(-1), reference.reshape(-1)

    RE_score = 1 - np.mean((mean_updated - reference)**2) / np.mean((mean_prior - reference)**2)
    return RE_score

def compute_CRPS(ensemble, reference, min_lat=-90, max_lat=90):
    """ Computes the continuous ranked probability score (CRPS).
    This scores evaluates how well a probabilistic forecast (given by an ensemble) 
    predicts a given reference. The CRPS is relative in the sense that it is used to 
    compare different forecasts, lower score being better.
    The CRPS is a sum of a misfit term and a spread term. Here both are returned separately.

    See Jordan et al., Evaluating Probabilistic Forecasts with scoringRule (2018).

    Parameters
    ----------
    ensemble: dask.array (n_members, m)
        Collection of prediction vectors.
    reference: dask.array (m)
        Ground truth to be reconstructed.
    min_lat: float, defaults to None.
        If specified, ignore the refions of low latitude 
        in the computation of the mismatch.
    max_lat: float, defaults to None.
        If specified, ignore the refions of high latitude 
        in the computation of the mismatch.

    Returns
    -------
    CRPS: dask.array (m)
        Vector of CRPS scores at each location.
    misfit: dask.array (m)
        Vector of misfits (in the CRPS) at each location.
    spread: dask.array (m)
        Vector of spreads (in the CRPS) at each location.

    """
    ensemble = ensemble[:, ~np.isnan(reference)]
    reference = reference[~np.isnan(reference)]

    n_members = ensemble.shape[0]
    misfit = (1 / n_members) * da.fabs(ensemble - reference.reshape(-1)[None, :]).sum(axis=0)
    spread = (1 / (2 * n_members**2)) * da.fabs(
            ensemble[None, :, :] - ensemble[:, None, :]).sum(axis=0).sum(axis=0)
    CRPS =  misfit -  spread
    return CRPS, misfit, spread

def compute_energy_score(ensemble, reference, min_lat=-90, max_lat=90):
    """ Computes energy score (multivariate generalisation of the CRPS".
    This scores evaluates how well a probabilistic forecast (given by an ensemble) 
    predicts a given reference. The energy score is relative in the sense that it is used to 
    compare different forecasts, lower score being better.
    The energy score is a sum of a misfit term and a spread term. Here both are returned separately.

    See Jordan et al., Evaluating Probabilistic Forecasts with scoringRule (2018).

    Parameters
    ----------
    ensemble: dask.array (n_members, m)
        Collection of prediction vectors.
    reference: dask.array (m)
        Ground truth to be reconstructed.
    min_lat: float, defaults to None.
        If specified, ignore the refions of low latitude 
        in the computation of the mismatch.
    max_lat: float, defaults to None.
        If specified, ignore the refions of high latitude 
        in the computation of the mismatch.

    Returns
    -------
    energy_score: dask.array (1)
        Energy score (scalar).
    misfit: dask.array (1)
        Misfit term of the energy score (scalar).
    spread: dask.array (1)
        Spread term of the energy score (scalar).

    """
    # Filter out high/low latitude if provided.
    lat_filter_inds = (reference.latitude < max_lat).data & (reference.latitude > min_lat).data
    reference = reference.data

    ensemble = ensemble[:, lat_filter_inds]
    reference = reference[lat_filter_inds]

    # Get rid of Nans.
    ensemble = ensemble[:, ~np.isnan(reference)]
    reference = reference[~np.isnan(reference)]

    n_members = ensemble.shape[0]
    misfit = (1 / n_members) * da.linalg.norm(
            ensemble - reference.reshape(-1)[None, :], axis=1).sum(axis=0)
    spread = (1 / (2 * n_members**2)) * da.linalg.norm(
            ensemble[None, :, :] - ensemble[:, None, :], axis=2).sum(axis=0).sum(axis=0)
    energy_score = misfit - spread
    return energy_score, misfit, spread

def compute_RMSE(mean_updated, reference, min_lat=-90, max_lat=90):
    """ Root mean square error.

    Parameters
    ----------
    mean_updated: dask.array (m)
        Vector of mean elements after updating.
    reference: dask.array (m)
        Ground truth to be reconstructed.
    min_lat: float, defaults to None.
        If specified, ignore the refions of low latitude 
        in the computation of the mismatch.
    max_lat: float, defaults to None.
        If specified, ignore the refions of high latitude 
        in the computation of the mismatch.

    Returns
    -------
    RMSE: float

    """
    # Filter out high/low latitude if provided.
    lat_filter_inds = (reference.latitude < max_lat).data & (reference.latitude > min_lat).data
    reference = reference.data

    mean_updated = mean_updated[lat_filter_inds]
    reference = reference[lat_filter_inds]

    # Get rid of Nans.
    mean_updated = mean_updated[~np.isnan(reference)]
    reference = reference[~np.isnan(reference)]

    # Make sure shapes agree.
    mean_updated, reference = mean_updated.reshape(-1), reference.reshape(-1)
    rmse = np.sqrt(np.mean((mean_updated - reference)**2))
    if isinstance(rmse, np.ndarray): rmse = rmse[0]

    return rmse
