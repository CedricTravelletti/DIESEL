""" Helper functions for the DIESEL package.

"""
import numpy as np
from numpy.core.numeric import array
from numpy import average

import dask.array as da
from dask.distributed import wait, progress


from builtins import CLIENT as client

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
        R = client.persist(R)
        wait(R)
        print("Cholesky result.")
        print(R.compute())
    except:
        print("Error in Cholesky")
        print(debug_string)
    try:
        R_inv = da.linalg.solve_triangular(R, da.linalg.eye(R.shape[0], chunks=chunk_size), lower=False)
        R_inv = client.persist(R_inv)
        wait(R_inv)
        print("Solve result.")
        print(R_inv.compute())
    except:
        print("Error in solve triangular")
        print(debug_string)

    # Extract the part of interest for us.
    if shape_diff > 0:
        R = R[:-shape_diff, :-shape_diff]
        R_inv = R_inv[:-shape_diff, :-shape_diff]
    return da.transpose(R), da.matmul(R_inv, da.transpose(R_inv))

def svd_invert(A, svd_rank=None):
    if svd_rank is None: svd_rank = A.shape[0]
    # Compute compressed SVD.
    # WARNING: dask return the already transposed version of v, 
    # so that A = u @ diag(s) @ v.
    # This is poorly documented in dask.
    u, s, v = da.linalg.svd_compressed(
                    A, k=svd_rank, compute=True) 
    # Compute (symmetric) square root.
    smat = da.diag(da.sqrt(s))
    sqrt = da.matmul(da.matmul(u, smat), da.transpose(u))

    imat = da.diag(da.true_divide(da.ones(s.shape), s))
    inv = da.matmul(da.matmul(da.transpose(v), imat), da.transpose(u))

    return sqrt, inv

def cross_covariance(X, Y, ddof=None, dtype=None):
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)

    X = array(X, ndmin=2, dtype=dtype)
    Y = array(Y, ndmin=2, dtype=dtype)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    avg_X, w_sum = average(X, axis=1, weights=w, returned=True)
    avg_Y, w_sum = average(Y, axis=1, weights=w, returned=True)

    fact = X.shape[1] - ddof

    # Subtract the mean.
    X -= avg_Y[:, None]
    Y -= avg_Y[:, None]
    
    Y_T = Y.T
    c = dot(X, Y_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()
