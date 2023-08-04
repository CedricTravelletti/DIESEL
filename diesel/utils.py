""" Helper functions for the DIESEL package.

"""
import numpy as np
from numpy import average
import torch

import dask.array as da
from dask.distributed import wait, progress
from dask.utils import apply, derived_from
from dask.array.core import (Array, asanyarray, asarray, blockwise, broadcast_arrays,
        broadcast_shapes, broadcast_to, concatenate, elemwise, from_array, implements,
        is_scalar_for_elemwise, map_blocks, stack, tensordot_lookup)
from dask.array.routines import array, dot


from climate.utils import match_vectors_indices


# Get the client stored in the global variable.
from builtins import CLIENT as global_client


CHUNK_REDUCTION_FACTOR = 4

def find_closest_multiple(x, base):
    closest_lower = int(base * round(float(x)/base))
    if closest_lower == x: return x
    else: return closest_lower + base

def cholesky_invert(A, debug_string):
    """ Computes the (lower) Cholesky factor and the inverse 
    of a symmetric positive definite matrix using Cholesky decomposition 
    and backward substitution.

    Parameters
    ----------
    A: dask.array

    Returns
    -------
    L, A_inv: dask.array (lazy)
        Lower Cholesky factor and inverse of the input matrix.

    """
    # Note that the daks cholesky implementation requires square chunks.
    # Hence, to keep chunks of a manageable size, one possible trick is to make 
    # R into a matrix wiht shape divisible by CHUNK_REDUCTION_FACTOR, to have CHUNK_REDUCTION_FACTOR chunks along each 
    # dimension.
    # Appending with identity matrix (in block diag fashion) allows us 
    # to recover the original Cholesky decomposition from the one of the 
    # augmented matrix.

    # If small enough then use only one chunk.
    if (A.shape[0] < CHUNK_REDUCTION_FACTOR - 1):
        chunk_size = A.shape[0]
        A_rechunked = A.rechunk(chunk_size)
        shape_diff = 0

    # If already square, then proceed.
    elif len(set(A.chunks[0] + A.chunks[1])) == 1: 
        A_rechunked = A
        shape_diff = 0
        chunk_size = A.chunks[0][0]

    # Else append identity matrix to get a shape that is 
    # divisible by CHUNK_REDUCTION_FACTOR.
    else:
        new_shape = find_closest_multiple(A.shape[0], CHUNK_REDUCTION_FACTOR)
        if new_shape > 0:
            shape_diff = new_shape - A.shape[0]
            A_rechunked = da.vstack(
                    [
                        da.hstack([A, da.zeros((A.shape[0], shape_diff))]),
                        da.hstack([da.zeros((shape_diff, A.shape[0])), da.eye(shape_diff)])
                    ]
            )
        chunk_size = int(new_shape / CHUNK_REDUCTION_FACTOR)
        A_rechunked = A_rechunked.rechunk(chunk_size)

    # TEMP: try to compute to see if fails.
    try:
        R = da.linalg.cholesky(A_rechunked, lower=False)
    except:
        print("Error in Cholesky")
        print(debug_string)
    try:
        R_inv = da.linalg.solve_triangular(R, da.linalg.eye(R.shape[0], chunks=chunk_size), lower=False)
    except:
        print("Error in solve triangular")
        print(debug_string)

    # Extract the part of interest for us.
    if shape_diff > 0:
        R = R[:-shape_diff, :-shape_diff]
        R_inv = R_inv[:-shape_diff, :-shape_diff]
    return da.transpose(R), da.matmul(R_inv, da.transpose(R_inv))

def svd_invert(A, svd_rank=None, client=global_client):
    if svd_rank is None: svd_rank = A.shape[0]
    # Compute compressed SVD.
    # WARNING: dask return the already transposed version of v, 
    # so that A = u @ diag(s) @ v.
    # This is poorly documented in dask.
    u, s, v = da.linalg.svd_compressed(
                    A, k=svd_rank, compute=True) 
    u, s, v = client.persist(u), client.persist(s), client.persist(v)

    # Compute (symmetric) square root.
    smat = da.diag(da.sqrt(s))
    sqrt = da.matmul(da.matmul(u, smat), da.transpose(u))

    imat = da.diag(da.true_divide(da.ones(s.shape), s))
    inv = da.matmul(da.matmul(da.transpose(v), imat), da.transpose(u))

    return sqrt, inv

def cross_covariance(X, Y, bias=False, ddof=None, dtype=None, rowvar=True):
    if not rowvar:
        X = X.T
        Y = Y.T
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    """
    if dtype is None:
        dtype = np.result_type(X, np.float64)

    X = array(X, ndmin=2, dtype=dtype)
    Y = array(Y, ndmin=2, dtype=dtype)
    """

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    avg_X = torch.mean(X, axis=1)
    avg_Y = torch.mean(Y, axis=1)

    fact = X.shape[1] - ddof

    # Subtract the mean.
    X -= avg_X[:, None]
    Y -= avg_Y[:, None]
    
    Y_T = Y.T
    c = torch.matmul(X, Y_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()

def build_forward_mean_per_cell(mean_ds, data_ds):
    """ Build the forward operator corresponding to a given 
    model grid and data point cloud. 
    This function only assimilated the mean observed value in each cell.

    Parameters
    ----------
    mean_ds: xr.DataArray
    data_ds: xr.DataArray

    Returns
    -------
    G_mean: (n_data_mean, n_cells)
        Forward operator for assimilation of mean data in each cell.
    mean_datas (n_data_mean)
        Vector of mean observed data in each cell.


    """
    # Get the model cell index corresponding to each observations.
    matched_inds = match_vectors_indices(mean_ds, data_ds)

    # Get unique indices. For the ones that appear several time, 
    # we will assimilat the mean. I.e. we assimilat the mean observed data 
    # in each cell where we have observations.
    unique_indices = np.unique(matched_inds)
    mean_datas = [np.mean(data_ds.values[matched_inds == i]) for i in unique_indices]
    median_datas = [np.median(data_ds.values[matched_inds == i]) for i in unique_indices]
    std_datas = [np.std(data_ds.values[matched_inds == i]) for i in unique_indices]
    n_datas = [len(data_ds.values[matched_inds == i]) for i in unique_indices]

    mean_datas, median_datas, std_datas, n_datas = np.array(mean_datas), np.array(median_datas), np.array(std_datas), np.array(n_datas)

    data_lats = mean_ds.latitude[unique_indices]
    data_lons = mean_ds.longitude[unique_indices]

    G = np.zeros((unique_indices.shape[0], mean_ds.shape[0]))
    for i in range(unique_indices.shape[0]):
        G[i, unique_indices[i]] = 1.0
    return G, mean_datas, std_datas, median_datas, n_datas, data_lons, data_lats

@derived_from(np)
def cov(m, y=None, rowvar=1, bias=0, ddof=None):
    """ Re-implementation of the dask.cov function. 
    The goal is to restrict the computation to float32 
    to save memory, apart from that, the implementation is the same. 

    """
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    m = asarray(m)
    if y is None:
        dtype = np.result_type(m, np.float32)
    else:
        y = asarray(y)
        dtype = np.result_type(m, y, np.float32)
    X = array(m, ndmin=2, dtype=dtype)

    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        N = X.shape[1]
        axis = 0
    else:
        N = X.shape[0]
        axis = 1

    # check ddof
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    fact = float(N - ddof)
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)
        fact = 0.0

    if y is not None:
        y = array(y, ndmin=2, dtype=dtype)
        X = concatenate((X, y), axis)

    X = X - X.mean(axis=1 - axis, keepdims=True)
    if not rowvar:
        return (dot(X.T, X.conj()) / fact).squeeze()
    else:
        return (dot(X, X.T.conj()) / fact).squeeze()
