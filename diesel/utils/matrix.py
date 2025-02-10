"""Matrix operations utilities."""

import numpy as np
import warnings
import dask.array as da
from dask.utils import derived_from
import torch


@derived_from(np)
def cov(m, y=None, rowvar=1, bias=0, ddof=None):
    """Re-implementation of the dask.cov function.
    The goal is to restrict the computation to float32
    to save memory, apart from that, the implementation is the same.

    """
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    m = da.asarray(m)
    if y is None:
        dtype = np.result_type(m, np.float32)
    else:
        y = da.asarray(y)
        dtype = np.result_type(m, y, np.float32)
    X = da.array(m, ndmin=2, dtype=dtype)

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
        y = da.array(y, ndmin=2, dtype=dtype)
        X = da.concatenate((X, y), axis)

    X = X - X.mean(axis=1 - axis, keepdims=True)
    if not rowvar:
        return (da.dot(X.T, X.conj()) / fact).squeeze()
    else:
        return (da.dot(X, X.T.conj()) / fact).squeeze()


def find_closest_multiple(x, base):
    closest_lower = int(base * np.round(float(x) / base))
    if closest_lower == x:
        return x
    else:
        return closest_lower + base


def cholesky_invert(A, debug_string, CHUNK_REDUCTION_FACTOR=4):
    """Computes the (lower) Cholesky factor and the inverse
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
    if A.shape[0] < CHUNK_REDUCTION_FACTOR - 1:
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
                    da.hstack([da.zeros((shape_diff, A.shape[0])), da.eye(shape_diff)]),
                ]
            )
        chunk_size = int(new_shape / CHUNK_REDUCTION_FACTOR)
        A_rechunked = A_rechunked.rechunk(chunk_size)

    # TEMP: try to compute to see if fails.
    try:
        R = da.linalg.cholesky(A_rechunked, lower=False)
    except np.linalg.LinAlgError:
        print("Error in Cholesky")
        print(debug_string)
    try:
        R_inv = da.linalg.solve_triangular(
            R, da.linalg.eye(R.shape[0], chunks=chunk_size), lower=False
        )
    except np.linalg.LinAlgError:
        print("Error in solve triangular")
        print(debug_string)

    # Extract the part of interest for us.
    if shape_diff > 0:
        R = R[:-shape_diff, :-shape_diff]
        R_inv = R_inv[:-shape_diff, :-shape_diff]
    return da.transpose(R), da.matmul(R_inv, da.transpose(R_inv))


def svd_invert(A, svd_rank=None, client=None):
    if client is None:
        # Get the client stored in the global variable.
        from builtins import CLIENT as client

    if svd_rank is None:
        svd_rank = A.shape[0]
    # Compute compressed SVD.
    # WARNING: dask return the already transposed version of v,
    # so that A = u @ diag(s) @ v.
    # This is poorly documented in dask.
    u, s, v = da.linalg.svd_compressed(A, k=svd_rank, compute=True)
    u, s, v = client.persist(u), client.persist(s), client.persist(v)

    # Compute (symmetric) square root.
    smat = da.diag(da.sqrt(s))
    sqrt = da.matmul(da.matmul(u, smat), da.transpose(u))

    imat = da.diag(da.true_divide(da.ones(s.shape), s))
    inv = da.matmul(da.matmul(da.transpose(v), imat), da.transpose(u))

    return sqrt, inv


def cross_covariance(X, Y, bias=False, ddof=None, dtype=None, rowvar=True):
    if not rowvar:
        X_in = X.T
        Y_in = Y.T
    else:
        X_in = X
        Y_in = Y
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

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

    avg_X = torch.mean(X_in, axis=1)
    avg_Y = torch.mean(Y_in, axis=1)

    fact = X_in.shape[1] - ddof

    # Subtract the mean.
    X_centred = X_in - avg_X[:, None]
    Y_centred = Y_in - avg_Y[:, None]

    Y_T = Y_centred.T
    c = torch.matmul(X_centred, Y_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()
