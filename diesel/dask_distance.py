""" Utilities to compute pairwise distances in Dask. 
        Copied from https://github.com/jakirkham/dask-distance

"""
import functools
import itertools

import numpy

import dask
import dask.array


def _asarray(a):
    """
    Creates a Dask array based on ``a``.

    Parameters
    ----------
    a : array-like
        Object to convert to a Dask Array.

    Returns
    -------
    a : Dask Array
    """

    if not isinstance(a, dask.array.Array):
        a = numpy.asarray(a)
        a = dask.array.from_array(a, a.shape)

    return a


def _atleast_2d(*arys):
    """
    Provide at least 2-D views of the arrays.

    Parameters
    ----------
    *arys : Dask Array
            Arrays to make at least 2-D

    Returns
    -------
    *res : Dask Array sequence
    """

    result = []
    for a in arys:
        a = _asarray(a)
        if a.ndim == 0:
            a = a[None, None]
        elif a.ndim == 1:
            a = a[None]

        result.append(a)

    if len(arys) == 1:
        result = result[0]

    return result


def _broadcast_uv(u, v):
    U = _atleast_2d(u)
    V = _atleast_2d(v)

    if U.ndim != 2:
        raise ValueError("u must be a 1-D or 2-D array.")
    if V.ndim != 2:
        raise ValueError("v must be a 1-D or 2-D array.")

    U = U[:, None]
    V = V[None, :]

    return U, V


def _unbroadcast_uv(u, v, result):
    u = _asarray(u)
    v = _asarray(v)

    if v.ndim == 1:
        result = result[:, 0]
    if u.ndim == 1:
        result = result[0]

    return result


def _broadcast_uv_wrapper(func):
    @functools.wraps(func)
    def _wrapped_broadcast_uv(u, v):
        U, V = _broadcast_uv(u, v)

        result = func(U, V)

        result = _unbroadcast_uv(u, v, result)

        return result

    return _wrapped_broadcast_uv


def _cdist_apply(U, V, metric):
    U = U[:, 0, :]
    V = V[0, :, :]

    result = numpy.empty((len(U), len(V)), dtype=float)

    for i, j in numpy.ndindex(result.shape):
        result[i, j] = metric(U[i], V[j])

    return result

def euclidean(u, v):
    """
    Finds the Euclidean distance between two 1-D arrays.

    .. math::

       \lVert u - v \\rVert_{2}

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       Euclidean distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = (abs(u - v) ** 2).sum(axis=-1) ** 0.5

    return result

def seuclidean(u, v, V):
    """
    Finds the standardized Euclidean distance between two 1-D arrays.

    .. math::

       \sqrt{\sum_{i} \left(
           \\frac{\left( u_{i} - v_{i} \\right)^{2}}{V_{i}}
       \\right)}

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays
        V:           1-D array of variances

    Returns:
        float:       standardized Euclidean
    """

    var = V
    del V

    var = _asarray(var)
    if var.ndim != 1:
        raise ValueError("V must have a dimension of 1.")

    U, V = _broadcast_uv(u, v)
    Var = var[None, None].repeat(U.shape[0], axis=0).repeat(U.shape[1], axis=1)

    U = U.astype(float)
    V = V.astype(float)
    Var = Var.astype(float)

    result = dask.array.sqrt(((abs(U - V) ** 2) / Var).sum(axis=-1))

    result = _unbroadcast_uv(u, v, result)

    return result

def cdist(XA, XB, metric="euclidean", **kwargs):
    """
    Finds the distance matrix using the metric on each pair of points.

    Args:
        XA:         2-D array of points
        XB:         2-D array of points
        metric:     string or callable
        **kwargs:   provided to the metric (see below)

    Keyword Args:
        p:          p-norm for minkowski only (default: 2)
        V:          1-D array of variances for seuclidean only
                    (default: estimated from XA and XB)
        VI:         Inverse of the covariance matrix for mahalanobis only
                    (default: estimated from XA and XB)
        w:          1-D array of weights for wminkowski only (required)

    Returns:
        array:      distance between each combination of points
    """

    func_mappings = {
        "braycurtis": braycurtis,
        "canberra": canberra,
        "chebyshev": chebyshev,
        "cityblock": cityblock,
        "correlation": correlation,
        "cosine": cosine,
        "dice": dice,
        "euclidean": euclidean,
        "hamming": hamming,
        "jaccard": jaccard,
        "kulsinski": kulsinski,
        "mahalanobis": mahalanobis,
        "minkowski": minkowski,
        "rogerstanimoto": rogerstanimoto,
        "russellrao": russellrao,
        "sokalmichener": sokalmichener,
        "sokalsneath": sokalsneath,
        "seuclidean": seuclidean,
        "sqeuclidean": sqeuclidean,
        "wminkowski": wminkowski,
        "yule": yule,
    }

    result = None
    if callable(metric):
        XA = _asarray(XA)
        XB = _asarray(XB)

        XA = XA.astype(float)
        XB = XB.astype(float)

        XA_bc, XB_bc = _broadcast_uv(XA, XB)

        XA_bc = XA_bc.rechunk(XA_bc.chunks[:-1] + ((XA_bc.shape[-1],),))
        XB_bc = XB_bc.rechunk(XB_bc.chunks[:-1] + ((XB_bc.shape[-1],),))

        result = dask.array.atop(
            _cdist_apply, "ij",
            XA_bc, "ijk",
            XB_bc, "ijk",
            dtype=float,
            concatenate=True,
            metric=metric
        )
    else:
        try:
            metric = metric.decode("utf-8")
        except AttributeError:
            pass

        metric = func_mappings[metric]

        if metric == mahalanobis:
            if "VI" not in kwargs:
                kwargs["VI"] = (
                    dask.array.linalg.inv(
                        dask.array.cov(dask.array.vstack([XA, XB]).T)
                    ).T
                )
        elif metric == minkowski:
            kwargs.setdefault("p", 2)
        elif metric == seuclidean:
            if "V" not in kwargs:
                kwargs["V"] = (
                    dask.array.var(dask.array.vstack([XA, XB]), axis=0, ddof=1)
                )
        elif metric == wminkowski:
            kwargs.setdefault("p", 2)

        result = metric(XA, XB, **kwargs)

    return result
