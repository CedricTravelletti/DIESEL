""" Helper functions for the DIESEL package.

"""
import dask.array as da


def cholesky_invert(A):
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
    R = da.linalg.cholesky(A, lower=False)
    R_inv = da.linalg.solve_triangular(R, da.linalg.eye(R.shape[0]), lower=False)
    return da.transpose(R), da.matmul(R_inv, da.transpose(R_inv))
